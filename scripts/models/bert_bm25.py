import time
import urllib

t1 = time.time()

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch import autograd
import numpy as np
import pescador
import string
import sys
import os
import argparse
from transformers.tokenization_distilbert import PRETRAINED_VOCAB_FILES_MAP
from pytorch_pretrained_bert import BertForTokenClassification
import random
random.seed(time.time())
from random import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from compute_statistics import *
from _bm25 import *
from training_utils import *

device = torch.device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
torch.manual_seed(33)
torch.set_printoptions(precision=6, threshold=100000)

parser = argparse.ArgumentParser()
parser.add_argument("--seen", action="store_true")
parser.add_argument("--fold", action="store_true")
parser.add_argument("--shr", action="store_true")
parser.add_argument("--ext", action="store_true")
parser.add_argument("--pat", action="store_true")
parser.add_argument("--hobby", action="store_true")
parser.add_argument("--profession", action="store_true")


args = parser.parse_args()

project_dir = str(Path(__file__).parent.parent.parent)
##########################  PARAMETERS  ##################################

# Policy params
bert_batch = 4
reduced_input_size = 2

# Optimizer params
batch_size = 2
reg_lambda = 2e-7
bert_lr = 0.00002
fc_lr = 0.001

predicate = "hobby"
predicate = "profession" if args.profession else "hobby"

##### Doc collection  #############

exp_name = "shr"
exp_name = "ext" if args.ext else exp_name
exp_name = "pat" if args.pat else exp_name

collection_path = project_dir + "/data/bert_documents/" + predicate + "_" + exp_name + ".txt"
collection_tfs = project_dir + "/data/bert_documents/" + predicate + "_" + exp_name + "_tfs.txt"

#############

# Input files
train_files_path = project_dir + "/data/" + predicate + "/datasets/train/"
train_files = [os.path.join(train_files_path, f) for f in os.listdir(train_files_path)]
test_file = project_dir + "/data/" + predicate + "/datasets/test.txt"
train_file = project_dir + "/data/" + predicate + "/datasets/train.txt"
predicate_path = project_dir + "/data/" + predicate + "/sources/" + predicate + "_list.txt"
predicate_list = [line.strip() for line in open(predicate_path, "r")]
input_vocab_path = PRETRAINED_VOCAB_FILES_MAP['vocab_file']['distilbert-base-uncased']
bert_vocab = [line.strip().decode("utf-8") for line in urllib.request.urlopen(input_vocab_path)]
input_vocab = [line.strip() for line in open("/home/tigunova/Documents/google-models/GoogleNews-vectors-negative300_vocab.txt")]

shorts = [x[0] for x in enumerate(input_vocab) if len(x[-1]) < 3]
shorts = torch.tensor(shorts).unsqueeze(-1).to(device)

############################################

# IR params
doc_len = 800
num_neg = 15
sample_n = 5
agg_type = "max"
k1 = 2.0
b = 0.75

# Training
num_epochs = 50
max_batch_epoch = 500 // batch_size

# pipeline
timesteps = 15

##################################### MODEL  ##################################

class RLPipeline(nn.Module):
    def __init__(self, **config):
        super(RLPipeline, self).__init__()
        self.ranker = bm25(df_dict=config["df_dict"], k1=k1, b=b, doc_num=len(config["doc_collection"]),
                           doc_len=doc_len)
        self.policy = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)
        self.policy.bert.encoder.layer = self.policy.bert.encoder.layer[:6]

##################################  Train and eval  ######################

import time
import math

def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)

def train(**config):
    # initialize components
    net = RLPipeline(**config)
    net.policy.to(device)

    bert_params = {'params': net.policy.bert.parameters(), 'lr':bert_lr}
    fc_params = {'params': net.policy.classifier.parameters(), 'lr': fc_lr}
    optimizer = torch.optim.Adam([bert_params, fc_params], weight_decay=reg_lambda)

    # input
    streams = [pescador.Streamer(feature_gen, ff, bert_batch) for ff in train_files]
    mux_stream = pescador.ShuffledMux(streams, random_state=33)

    # batch arrays
    docs_list = []
    doc_labels_list = []
    doc_ids_list = []
    labels = []
    logits = []
    words = []
    ranker_words = []

    # Train the Model
    for epoch in range(num_epochs):
        print("Epoch " + str(epoch))
        t1 = time.time()
        for i, train_features in enumerate(mux_stream):
            train_features = train_features[:reduced_input_size]  # reduce the input so that it fits

            ######## Run policy
            curr_words = torch.tensor([f["input_ids"] for f in train_features], dtype=torch.long).to(device)
            curr_mask = torch.tensor([f["input_mask"] for f in train_features], dtype=torch.long).to(device)
            curr_hash = torch.tensor([f["hash_mask"] for f in train_features], dtype=torch.long).to(device)
            curr_ranker_words = torch.tensor([f["ranker_tokens"] for f in train_features], dtype=torch.long).to(device)
            curr_segments = torch.zeros_like(curr_mask).to(device)
            label = train_features[0]["label_ids"]

            curr_hash = torch.where(
                curr_ranker_words.view(1, -1).eq(shorts).sum(0).view(reduced_input_size, -1).type(torch.ByteTensor).to(device),
                torch.zeros_like(curr_hash), curr_hash)

            one_logit, _ = net.policy(curr_words, curr_segments, curr_mask)

            # Mask
            multiplier = curr_mask.float() * curr_hash.float()
            one_logit = one_logit.squeeze()
            one_logit = torch.where(multiplier != 0, one_logit, torch.ones_like(one_logit) * np.NINF)
            one_logit = torch.where(curr_ranker_words != 0, one_logit, torch.ones_like(one_logit) * np.NINF)
            one_logit = one_logit.view((1, -1)).squeeze()
            curr_words = curr_words.view((1, -1)).squeeze()
            curr_ranker_words = curr_ranker_words.view((1, -1)).squeeze()

            labels.append(label)
            logits.append(one_logit)
            words.append(curr_words)
            ranker_words.append(curr_ranker_words)

            ######## Prepare documents
            curr_docs, curr_labels, doc_ids = sample_advanced(config["doc_dict"], label, num_neg, sample_n, add_tfs = True)
            docs_list.append(curr_docs)
            doc_labels_list.append(curr_labels)
            doc_ids_list.append(doc_ids)

            # When the batch is accumulated
            if (i+1) % batch_size == 0:
                optimizer.zero_grad()

                ## Concat all arrays
                logits = torch.stack(logits, dim=0)
                words = torch.stack(words, dim=0)
                ranker_words = torch.stack(ranker_words, dim=0)

                # Arrays by timestep
                total_query = []
                total_bert_query = []
                total_indexes = []
                total_logits = []
                total_reward = []
                loss = 0

                for t in range(timesteps):
                    # sample a word
                    indices = Categorical(logits=logits).sample().unsqueeze(1)
                    total_indexes.append(indices)
                    logits_sm = logits.softmax(dim=1)
                    taken_logit = (logits_sm.gather(1, indices).log()).squeeze(1)#.mean(dim=1).to(device) #suspicious

                    total_query.append(ranker_words.gather(1, indices))
                    total_bert_query.append(words.gather(1, indices))

                    total_logits.append((logits.gather(1, indices)).squeeze(1))
                    query = torch.stack(total_query, dim=1).squeeze(2)
                    bert_query = torch.stack(total_bert_query, dim=1).squeeze(2)

                    # remove the sampled element
                    maskk = torch.ones_like(logits)
                    maskk.scatter_(1, indices, 0.)
                    logits = logits * maskk
                    logits = torch.where(logits != 0, logits, torch.ones_like(logits) * np.NINF)

                    ##### Run ranker
                    # for each subject in a batch
                    batch_reward = []
                    batch_scores = []
                    for answer, curr_query, curr_docs, curr_labels, curr_docid, curr_bert_query in zip(labels, query, docs_list, doc_labels_list, doc_ids_list, bert_query):
                        # for each doc execute query
                        scores_list = []
                        df = [net.ranker.df_dict.get(term.item(), 0) for term in curr_query]
                        idf = torch.tensor(
                            [math.log((net.ranker.doc_num - termdf + 0.5) / (termdf + 0.5)) for termdf in df])#.to(device)
                        for doc, did in zip(curr_docs, curr_docid):
                            qtf = torch.FloatTensor(
                                [tfs_dict[did].get(term.item(), 0) for term in curr_query])#.to(device)
                            scores_list.append(net.ranker(qtf, idf, doc))
                        scores_list = torch.stack(scores_list).to(device)

                        # get ranks
                        scores_list = aggregate_scores(scores_list, curr_labels, len(predicate_list), agg_type)
                        scores_list = torch.where(scores_list != 0, scores_list, torch.ones_like(scores_list) * np.NINF)
                        batch_scores.append(scores_list)
                        sorted_list, sorted_idx = torch.sort(scores_list, dim=0, descending=True)
                        answer = torch.tensor(answer).to(device)

                        ranks = isin(sorted_idx, answer).nonzero() + 2
                        dcg = torch.sum(1.0 / torch.log2(ranks.float()))
                        id_dcg = torch.sum(1.0 / torch.log2(torch.FloatTensor([x+2 for x in range(ranks.size(0))])))
                        ndcg = dcg / id_dcg

                        batch_reward.append(ndcg)

                    batch_reward = torch.tensor(batch_reward).to(device)#.stack(batch_reward, dim=0)
                    total_reward.append(batch_reward)
                    loss += -torch.dot(batch_reward, taken_logit) / batch_size

                loss /= timesteps
                loss.backward()
                optimizer.step()

                # batch arrays
                docs_list = []
                doc_labels_list = []
                doc_ids_list = []
                labels = []
                logits = []
                words = []
                ranker_words = []

                if i // batch_size > max_batch_epoch:
                    break

        print("one epoch time ", time.time() - t1)

    # save model
    torch.save(net.state_dict(), model_path)

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1]#.sort(dim=dim)[0]
    return index#x.gather(dim, index)

def validate(output_path = None, **config):
    with torch.no_grad():
        # initialize components
        global output_file
        net = RLPipeline(**config)
        net.policy.to(device)
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
        net.eval()

        output_file = output_path if output_path != None else output_file

        # input
        streamer = pescador.Streamer(feature_gen, test_file, 4)
        with open(output_file, "w") as f_test_out:
            ctr = 0
            for train_features in streamer:
                ctr += 1
                train_features = train_features[:reduced_input_size] # reduce the input so that it fits

                ######## Run policy
                curr_words = torch.tensor([f["input_ids"] for f in train_features], dtype=torch.long).to(device)
                curr_mask = torch.tensor([f["input_mask"] for f in train_features], dtype=torch.long).to(device)
                curr_hash = torch.tensor([f["hash_mask"] for f in train_features], dtype=torch.long).to(device)
                curr_ranker_words = torch.tensor([f["ranker_tokens"] for f in train_features], dtype=torch.long).to(device)
                curr_segments = torch.zeros_like(curr_mask).to(device)

                one_logit, _ = net.policy(curr_words, curr_segments, curr_mask)

                curr_hash = torch.where(
                    curr_ranker_words.view(1, -1).eq(shorts).sum(0).view(reduced_input_size, -1).type(torch.ByteTensor).to(
                        device),
                    torch.zeros_like(curr_hash), curr_hash)

                # Mask
                multiplier = curr_hash.float() * curr_mask.float()
                one_logit = one_logit.squeeze()
                one_logit = torch.where(multiplier != 0, one_logit, torch.ones_like(one_logit) * np.NINF)
                one_logit = torch.where(curr_ranker_words != 0, one_logit, torch.ones_like(one_logit) * np.NINF)
                one_logit = one_logit.view((1, -1))
                curr_ranker_words = curr_ranker_words.view((1, -1))

                # Select kmax query words
                idxs = kmax_pooling(one_logit, 1, timesteps)
                query = curr_ranker_words.gather(1, idxs)[0]

                # get ranking
                curr_docs = torch.stack(doc_tensors).squeeze()
                curr_labels = torch.LongTensor(doc_labels)

                scores_list = []
                df = [net.ranker.df_dict.get(term.item(), 0) for term in query]
                idf = torch.tensor([math.log((net.ranker.doc_num - termdf + 0.5) / (termdf + 0.5)) for termdf in df])#.to(device)
                for num, doc in zip(doc_ids, curr_docs):
                    qtf = torch.FloatTensor([tfs_dict[num].get(term.item(), 0) for term in query])#.to(device)
                    scores_list.append(net.ranker(qtf, idf, doc))
                scores_list = torch.stack(scores_list).to(device)
                scores_list = aggregate_scores(scores_list, curr_labels, len(predicate_list), agg_type)
                scores_list = torch.where(scores_list != 0, scores_list, torch.ones_like(scores_list) * np.NINF)
                scores_list = scores_list.cpu().data.numpy()

                f_test_out.write(str(train_features[0]["guid"]) + "\t" + repr(train_features[0]["label_ids"]) + '\t' + '\t'.join(
                        [str(y) for y in sorted(enumerate(scores_list), key=lambda x: x[1], reverse=True)]) + '\n')

        stats = compute_whatever_stats(output_file)
        print(stats)
        return stats

##########################################

####################################  MAIN  #####################################################################

doc_tensors, doc_labels, real_docs, df_dict, doc_dict, tfs_dict, doc_ids = load_documents(collection_path, predicate_list, doc_len=doc_len, tfs_file = collection_tfs)

config = dict(input_weights=None, doc_collection=doc_tensors, doc_dict=doc_dict, df_dict=df_dict,
              normalized_weights=None, dim=768)

##################################################################################

###################################  SEEN  #####################################################

if args.seen:
    filename = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])
    model_path = project_dir + "/data/tmp_folder/" + filename + ".pkl"
    seen_dir = project_dir + "/data/" + predicate + "/results/" + exp_name + "/output_files/seen/"
    Path(seen_dir).mkdir(parents=True, exist_ok=True)
    output_file =  seen_dir + "bert_bm25.txt"
    train(**config)
    os.remove(model_path)
    exit(0)

##############################   FOLDS    #####################################################################

if args.fold:
    produce_queries = False
    num_folds = 10
    timesteps = 10
    agg_type = "max"
    num_epochs = 10
    filename = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])
    model_path = project_dir + "/data/tmp_folder/" + filename + ".pkl"
    grid_dir = project_dir + "/data/" + predicate + "/results/" + exp_name + "/"
    Path(grid_dir).mkdir(parents=True, exist_ok=True)
    output_dir = project_dir + "/data/" + predicate + "/results/" + exp_name + "/output_files/folds/bert_bm25"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(grid_dir + "folds_bert_bm25.txt", "w") as f_resu:
        for i in range(num_folds):
            output_file = os.path.join(output_dir, str(i) + ".txt")
            print("fold number ", i)
            train_files_path = project_dir + "/data/" + predicate + "/folds_datasets/" + str(i) + "/train/"
            train_files = [os.path.join(train_files_path, f) for f in os.listdir(train_files_path)]
            test_file = project_dir + "/data/" + predicate + "/folds_datasets/" + str(i) + "/test.txt"
            train(**config)
            stats = validate(**config)
            f_resu.write("%s\t%s\n" % (str(i), repr(stats)))
            f_resu.flush()
    os.remove(model_path)
    exit(0)
