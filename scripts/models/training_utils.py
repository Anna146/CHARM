import csv
import pickle
import random

from collections import defaultdict
from numpy.linalg import norm

import numpy as np
import sys

import torch

import pickle

default_device = torch.device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

class InputFeatures(object):
    def __init__(self, input_ids = None, input_mask = None, segment_ids = None, label_ids = None, guid = None,
                 hash_mask = None, ranker_tokens = None, plain_texts = None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.hash_mask = hash_mask
        self.label_ids = label_ids
        self.guid = guid
        self.ranker_tokens = ranker_tokens
        self.plain_texts = plain_texts

    def save(self, file):
        with open(file, "ab") as f:
            pickle.dump(self.__dict__, f)

    def load(self, f):
        self.__dict__ = pickle.load(f)


def feature_gen(filepath, batch_size):
    with open(filepath, "rb") as f_in:
        while True:
            batch = []
            sample = InputFeatures()
            try:
                sample.load(f_in)
            except (EOFError, pickle.UnpicklingError):
                raise StopIteration
            for inp_id, inp_msk, rank_tok, hash_msk in zip(np.split(np.array(sample.input_ids), batch_size),
                                                         np.split(np.array(sample.input_mask), batch_size),
                                                         np.split(np.array(sample.ranker_tokens), batch_size),
                                                         np.split(np.array(sample.hash_mask), batch_size)):
                batch.append(dict(input_ids=inp_id,
                              input_mask=inp_msk,
                              ranker_tokens=rank_tok,
                              label_ids=sample.label_ids,
                              guid=sample.guid,
                              hash_mask=hash_msk
                            ))
            yield batch


def sample_advanced(doc_dict, label, num_neg, sample_n, add_tfs = False):
    sampled_tensors = []
    sampled_labels = []
    sampled_docs = []
    sampled_ids = []
    take_only = random.sample(doc_dict.keys(), k=min(num_neg, len(doc_dict)))
    take_only = take_only + label
    for k, v in doc_dict.items():
        if k in take_only:
            curr_sample = min(sample_n, len(v))
            sampled_pairs = random.sample(v, k=curr_sample)
            sampled_tensors.extend([x[0] for x in sampled_pairs])
            sampled_docs.extend([x[1] for x in sampled_pairs])
            sampled_ids.extend([int(x[1].split(":::")[-1]) for x in sampled_pairs])
            sampled_labels.extend([k] * curr_sample)
    if add_tfs:
        return torch.stack(sampled_tensors, dim=0).squeeze(1), torch.LongTensor(sampled_labels), sampled_ids
    else:
        return torch.stack(sampled_tensors, dim=0).squeeze(1), torch.LongTensor(sampled_labels)

def load_weights(path):
    input_weights = np.load(path)
    vocab_len = input_weights.shape[0]
    input_weights = np.append(input_weights.astype(float), np.zeros(shape=(1, input_weights.shape[1])), axis=0)
    return input_weights, vocab_len

def load_weights_normalized(path):
    input_weights = np.load(path).astype(float)
    nor = np.expand_dims(norm(input_weights, axis=1), 1)
    input_weights /= (nor + 1e-7)
    vocab_len = input_weights.shape[0]
    input_weights = np.append(input_weights, np.zeros(shape=(1, input_weights.shape[1])), axis=0)
    return input_weights, vocab_len

def load_documents(documents, predicate_list, doc_len = 1000, removes = None, tfs_file = None):
    doc_tensors = []
    doc_labels = []
    doc_ids = []
    doc_dict = defaultdict(list)
    real_docs = []
    df_dict = {}
    tfs_dict = {}
    if tfs_file != None:
        tfs_dict = pickle.load(open(tfs_file, "rb"))
    with open(documents, "r") as f_in:
        reader = csv.reader(f_in)
        for line in reader:
            if line[0] not in predicate_list:
                continue
            curr_label = predicate_list.index(line[0])
            doc_labels.append(curr_label)
            real_docs.append(line[-1][:3000])
            doc_ids.append(int(line[-1].split(":::")[-1]))
            if removes == None:
                doc = [int(x) for x in line[1:-1]][:doc_len]
            else:
                doc = [int(x) for x in line[1:-1] if x not in removes][:doc_len]
            if len(doc) < doc_len:
                doc.extend([-1] * (doc_len - len(doc)))
            for tok in set(doc):
                df_dict[tok] = df_dict.get(tok, 0.0) + 1
            doc_tensor = torch.LongTensor([doc])
            doc_tensors.append(doc_tensor)
            doc_dict[curr_label].append((doc_tensor, real_docs[-1]))
    return doc_tensors, doc_labels, real_docs, df_dict, doc_dict, tfs_dict, doc_ids

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return index

def aggregate_scores(scores_list, label_list, predicate_num, agg_type, kavg = 5, device = default_device):
    result = torch.zeros(predicate_num).to(device)
    for l in torch.unique(label_list):
        curr_pred = scores_list[label_list == l]
        if curr_pred.size()[0] > 0:
            if agg_type == "avg":
                result[l] = curr_pred.sum() / curr_pred.size()[0]
            elif agg_type == "max":
                result[l] = torch.max(curr_pred)
            elif agg_type == "kavg":
                llen = min(curr_pred.size()[0], kavg)
                ind = kmax_pooling(curr_pred, k=llen, dim=0)
                result[l] = curr_pred.gather(0, ind).sum() / llen
    return result


def load_doublebert_documents(documents, device = "cpu"):
    doc_dict = defaultdict(list)
    with open(documents, "rb") as f_in:
        while True:
            sample = InputFeatures()
            try:
                sample.load(f_in)
            except (EOFError, pickle.UnpicklingError):
                break
            #doc_tensors.append((sample.input_ids, sample.segment_ids, sample.input_mask))
            input_ids = torch.tensor(sample.input_ids, dtype=torch.long).to(device)
            doc_dict[sample.label_ids[0]].append(([input_ids,
                                         torch.zeros_like(input_ids).to(device),
                                         torch.tensor(sample.input_mask, dtype=torch.long).to(device)], sample.plain_texts))
    return dict(doc_dict)

def sample_advanced_bert(doc_dict, label, num_neg, sample_n):
    sampled_tensors = []
    sampled_labels = []
    sampled_docs = []
    take_only = random.sample(doc_dict.keys(), k=min(num_neg, len(doc_dict)))
    take_only = take_only + label
    for k, v in doc_dict.items():
        if k in take_only:
            curr_sample = min(sample_n, len(v))
            sampled_pairs = random.sample(v, k=curr_sample)
            sampled_tensors.extend([x for x in sampled_pairs])
            sampled_labels.extend([k] * curr_sample)
    return sampled_tensors, sampled_labels

def sample_one_doc(doc_dict, labels, pos_prob=0.5):
    if random.random() > 1.0 - pos_prob: # sample pos
        sampled_cand = random.choice(labels)
    else: # sample neg
        sampled_cand = random.choice([l for l in doc_dict.keys() if l not in labels])
    return sampled_cand, random.choice(doc_dict[sampled_cand])

import random

def pos_and_neg_aggregation(scores_list, label_list, true_lab, agg_type):
    pos = []
    neg = []
    zipped = list(zip(scores_list, label_list))
    del scores_list
    del label_list
    #random.shuffle(zipped)
    #print(true_lab, zipped)
    for sc, lab in zipped:
        if lab in true_lab:
            pos.append(sc)
        else:
            neg.append(sc)
    pos = torch.stack(pos)
    neg = torch.stack(neg)
    if agg_type == "avg":
        pos = pos.sum() / pos.size()[0]
        neg = neg.sum() / neg.size()[0]
    elif agg_type == "max":
        pos = torch.max(pos)
        neg = torch.max(neg)
    return pos, neg


# expecting format [true_label, [word_indexes]]
def queries_gen(filepath, batch_size, word_num):
    with open(filepath, "r") as f_in:
        batch_X = np.empty(shape=(batch_size, word_num), dtype=int)
        while True:
            batch_y = []
            i = 0
            while i < batch_size:
                data = f_in.readline().strip()
                if data == "":
                    raise StopIteration()
                data = data.split(",")
                data_x = data[1].split(" ")[:word_num]
                if len(data_x) < word_num:
                    data_x = data_x + [-1] * (word_num - len(data_x))
                batch_X[i] = data_x
                batch_y.append([int(y) for y in data[0].split(" ")])
                i += 1
            yield np.array(batch_X), np.array(batch_y)

def sample_pos_neg(doc_dict, label_stack):
    pos_stack = []
    neg_stack = []
    for labels in label_stack:
        sampled_pos = random.choice(labels)
        sampled_neg = random.choice([l for l in doc_dict.keys() if l not in labels])
        pos_stack.append(random.choice(doc_dict[sampled_pos])[0])
        neg_stack.append(random.choice(doc_dict[sampled_neg])[0])
    stack = torch.stack(pos_stack + neg_stack, dim=0).squeeze()
    return stack