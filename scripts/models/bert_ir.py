import string
import time
import urllib

t1 = time.time()

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch import autograd
import numpy as np
import pescador
import sys
import os
import argparse
from transformers.tokenization_distilbert import PRETRAINED_VOCAB_FILES_MAP
from pytorch_pretrained_bert import BertModel
from random import random

sys.path.insert(0, str(Path(__file__).parent.parent))
from compute_statistics import *
from training_utils import *

torch.manual_seed(33)
torch.set_printoptions(precision=6, threshold=100000)


parser = argparse.ArgumentParser()
parser.add_argument("--shr", action="store_true")
parser.add_argument("--ext", action="store_true")
parser.add_argument("--pat", action="store_true")
parser.add_argument("--hobby", action="store_true")
parser.add_argument("--profession", action="store_true")

parser.add_argument('--device', type=int, default=None)

args = parser.parse_args()

project_dir = str(Path(__file__).parent.parent.parent)

device_string = "cuda:0"
if args.device != None:
    device_string = "cuda:" + str(args.device)

device = torch.device(torch.device(device_string if torch.cuda.is_available() else "cpu"))
cpu_device = torch.device(torch.device("cpu"))


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

##### Doc collection  #############t

exp_name = "shr"
exp_name = "ext" if args.ext else exp_name
exp_name = "pat" if args.pat else exp_name

collection_path = project_dir + "/data/double_bert_documents/" + predicate + "_" + exp_name + ".txt"
doc_dict = load_doublebert_documents(collection_path, device="cpu")

#############

# Input files
train_files_path = project_dir + "/data/" + predicate + "/datasets/train/"
train_files = [os.path.join(train_files_path, f) for f in os.listdir(train_files_path)]
test_file = project_dir + "/data/" + predicate + "/datasets/test.txt"
predicate_path = project_dir + "/data/" + predicate + "/sources/" + predicate + "_list.txt"
predicate_list = [line.strip() for line in open(predicate_path, "r")]
input_weights_path = project_dir + "/data/embeddings/GoogleNews-vectors-negative300.npy"
input_vocab_path = PRETRAINED_VOCAB_FILES_MAP['vocab_file']['distilbert-base-uncased']
bert_vocab = [line.strip().decode("utf-8") for line in urllib.request.urlopen(input_vocab_path)]
input_vocab = [line.strip() for line in open(project_dir + "/data/embeddings/GoogleNews-vectors-negative300_vocab.txt")]

sep_index = bert_vocab.index('[SEP]')

############################################

# IR params
bert_size = 6

# Training
iter_batch_size = 8
num_epochs = 200
max_batch_epoch = 50 // batch_size

hidden_size = 768

##################################### MODEL  ##################################

class BertMetric(nn.Module):
    def __init__(self, net_device, **config):
        super(BertMetric, self).__init__()
        self.bert_utt = BertModel.from_pretrained('bert-base-uncased').to(net_device)
        self.bert_utt.encoder.layer = self.bert_utt.encoder.layer[:bert_size]
        self.fc = nn.Linear(hidden_size, 1).to(net_device)

    def forward(self, x, y, z):
        return self.fc(self.bert_utt(x, y, z)[1])

##################################  Train and eval  ######################

import copy
from collections import defaultdict

def train(**config):
    with torch.cuda.device(device_string):
        torch.cuda.empty_cache()
    # initialize components
    net = BertMetric(net_device=device, **config)
    bert_params = {'params': net.bert_utt.parameters(), 'lr': bert_lr}

    optimizer = torch.optim.Adam([bert_params], lr=bert_lr, weight_decay=reg_lambda)
    criterion = torch.nn.BCELoss()
    # input
    streams = [pescador.Streamer(feature_gen, ff, bert_batch) for ff in train_files]
    mux_stream = pescador.ShuffledMux(streams, random_state=33)

    doc_lens = defaultdict(int)

    # Train the Model
    for epoch in range(num_epochs):
        print("Epoch " + str(epoch))
        for i, train_features in enumerate(mux_stream):
            ######## Run utterance bert
            train_features = train_features[:reduced_input_size]
            curr_words = torch.tensor([f["input_ids"] for f in train_features], dtype=torch.long).to(device1)
            curr_mask = torch.tensor([f["input_mask"] for f in train_features], dtype=torch.long).to(device1)
            curr_segments = torch.zeros_like(curr_mask).to(device1)
            label = train_features[0]["label_ids"]
            curr_words = torch.stack([x for x in curr_words.view(reduced_input_size * 2, -1) if x[0] != 0])
            curr_mask = torch.stack([x for x in curr_mask.view(reduced_input_size * 2, -1) if x[0] != 0])
            curr_segments = curr_segments.view(reduced_input_size * 2, -1)
            curr_segments = torch.stack([curr_segments.view(reduced_input_size * 2, -1)[ii] for ii in range(len(curr_words))])

            ######## Run documents bert
            curr_label, curr_doc = sample_one_doc(doc_dict, label)
            curr_doc = curr_doc[0]
            for jj in range(len(curr_doc)):
                if jj == 1:
                    curr_doc[jj] = torch.ones_like(curr_doc[jj])
                curr_doc[jj] = curr_doc[jj].view(-1, 256)[:2*reduced_input_size].to(device)
                curr_doc[jj] = torch.stack([curr_doc[jj][ii] for ii in range(len(curr_doc[0])) if curr_doc[0][ii][0] != 0])
            new_inp = []
            ########
            # a is utt and b is doc
            # position in a is div(nnn)
            # position in b is mod(nnn)
            iin = 0
            for a, b in zip([curr_words, curr_segments, curr_mask], curr_doc):
                mm, nnn = a.size()[0], b.size()[0]
                a = a.repeat(1, nnn).view(mm * nnn, -1)
                if iin == 0:
                    c = torch.ones((mm * nnn, 1), dtype=torch.long).to(device) * sep_index
                    a = torch.cat([a[:,:-1],c], dim=1)
                b = b.repeat(mm, 1)
                new_inp.append(torch.stack([a, b], dim = 1).view(mm * nnn, -1))
                iin += 1
            doc_lens[nnn] += 1
            intrm = torch.sum(net(*new_inp), dim=0) / mm / nnn
            curr_score = nn.functional.sigmoid(intrm)
            losses = criterion(curr_score.to(cpu_device), torch.FloatTensor([int(curr_label in label)]))
            losses.backward()
            with torch.cuda.device(device_string):
                torch.cuda.empty_cache()

            # When the batch is accumulated
            if (i+1) % iter_batch_size == 0:
                losses /= iter_batch_size
                optimizer.step()
                optimizer.zero_grad()
                del losses
                del curr_doc
                with torch.cuda.device(device_string):
                    torch.cuda.empty_cache()

                if i // batch_size > max_batch_epoch:
                    break

    # save model
    torch.save(net.state_dict(), model_path)

test_split_size = 128

from torch.nn.utils.rnn import pad_sequence

def sample_subset(doc_repr_dict, sample_size = 2):
    label_list = []
    doc_list = []
    real_list = []
    for lab, docs in doc_repr_dict.items():
        sampled = random.sample(docs, k=min(len(docs), sample_size))
        doc_list.extend([z[0] for z in sampled])
        real_list.extend([z[1] for z in sampled])
        label_list.extend([lab] * min(len(docs), sample_size))
    return label_list, doc_list, real_list

def validate(output_path = None, **config):
    t2 = time.time()
    with torch.no_grad():
        global output_file
        # initialize components
        net = BertMetric(net_device=device2, **config)
        net.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
        net.eval()

        output_file = output_path if output_path != None else output_file

        subset_size = 3

        # input
        streamer = pescador.Streamer(feature_gen, test_file, 4)

        doc_repr_dict = dict()
        for lab, docs in doc_dict.items():
            doc_repr_dict[lab] = []
            for curr_doc, real in docs:
                for jj in range(len(curr_doc)):
                    if jj == 1:
                        curr_doc[jj] = torch.ones_like(curr_doc[jj])
                    curr_doc[jj] = curr_doc[jj].view(-1, 256)[:2 * reduced_input_size].to(device2)
                    curr_doc[jj] = torch.stack([curr_doc[jj][ii] for ii in range(len(curr_doc[0])) if curr_doc[0][ii][0] != 0])
                doc_repr_dict[lab].append((curr_doc, real))

        utt_lens = defaultdict(int)
        doc_lens = defaultdict(int)

        for agg_type in ["max"]:
            with open(output_file, "w") as f_test_out:
                ctr = 0
                for train_features in streamer:
                    ctr += 1
                    train_features = train_features[:reduced_input_size] # reduce the input so that it fits

                    ######## Run policy
                    curr_words = torch.tensor([f["input_ids"] for f in train_features], dtype=torch.long).to(device2)
                    curr_mask = torch.tensor([f["input_mask"] for f in train_features], dtype=torch.long).to(device2)
                    curr_segments = torch.zeros_like(curr_mask).to(device2)
                    curr_words = torch.stack([x for x in curr_words.view(reduced_input_size * 2, -1) if x[0] != 0])
                    curr_mask = torch.stack([x for x in curr_mask.view(reduced_input_size * 2, -1) if x[0] != 0])
                    curr_segments = curr_segments.view(reduced_input_size * 2, -1)
                    curr_segments = torch.stack(
                        [curr_segments.view(reduced_input_size * 2, -1)[ii] for ii in range(len(curr_words))])

                    labels_list, docs_list, titles_list = sample_subset(doc_repr_dict, sample_size=subset_size)

                    inp_list = []
                    split_sizes = []
                    all_mm = []
                    all_nnn = []
                    for curr_doc in docs_list:
                        new_inp = []
                        iin = 0
                        for a, b in zip([curr_words, curr_segments, curr_mask], curr_doc):
                            mm, nnn = a.size()[0], b.size()[0]
                            a = a.repeat(1, nnn).view(mm * nnn, -1)
                            if iin == 0:
                                c = torch.ones((mm * nnn, 1), dtype=torch.long).to(device) * sep_index
                                a = torch.cat([a[:, :-1], c], dim=1)
                            b = b.repeat(mm, 1)
                            new_inp.append(torch.stack([a, b], dim=1).view(mm * nnn, -1))
                            iin += 1
                        all_mm.append(mm)
                        all_nnn.append(nnn)
                        utt_lens[mm] += 1
                        doc_lens[nnn] += 1
                        del a
                        del b
                        inp_list.append(new_inp)
                        split_sizes.append(new_inp[0].size()[0])
                    for ij in range(3):
                        new_inp[ij] = torch.split(torch.cat([xx[ij] for xx in inp_list], dim=0), split_size_or_sections=test_split_size, dim=0)
                    del inp_list
                    del curr_words
                    del curr_segments
                    del curr_mask
                    scores_stack = []
                    for ij in range(len(new_inp[0])):
                        scores_stack.append(net(new_inp[0][ij],new_inp[1][ij],new_inp[2][ij]).squeeze())
                    scores_stack = [xx if len(xx.size()) > 0 else xx.unsqueeze(0) for xx in scores_stack]
                    scores_stack = torch.split(torch.cat(scores_stack, dim=0), split_size_or_sections=split_sizes, dim=0)
                    scores_stack = pad_sequence(scores_stack, batch_first=True, padding_value = 0.0)
                    scores_list = torch.sum(scores_stack, dim=1) / torch.FloatTensor(split_sizes).to(device2)

                    del scores_stack
                    scores_list = aggregate_scores(scores_list, torch.tensor(labels_list), len(predicate_list), agg_type, device = device)
                    scores_list = scores_list.cpu().data.numpy()

                    f_test_out.write(
                        str(train_features[0]["guid"]) + "\t" + repr(train_features[0]["label_ids"]) + '\t' + '\t'.join(
                            [str(y) for y in sorted(enumerate(scores_list), key=lambda x: x[1], reverse=True)]) + '\n')
                    f_test_out.flush()
            stats = compute_whatever_stats(output_file)
            stats["aggregation"] = agg_type
            print(stats)
        return stats

##############################   FOLDS    #####################################################################

config = dict(input_weights=None, doc_dict=doc_dict, dim=768)

num_folds = 10
timesteps = 10
agg_type = "avg"
num_epochs = 200
filename = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])
model_path = project_dir + "/data/tmp_folder/" + filename + ".pkl"
grid_dir = project_dir + "data/" + predicate + "/results/" + exp_name + "/"
Path(grid_dir).mkdir(parents=True, exist_ok=True)
output_dir = project_dir + "data/" + predicate + "/results/" + exp_name + "/output_files/folds/bert_baseline"
Path(output_dir).mkdir(parents=True, exist_ok=True)
with open(grid_dir + "folds_bert_baseline.txt", "w") as f_resu:
    for i in range(num_folds - 1, 0, -1):
        output_file = os.path.join(output_dir, str(i) + ".txt")
        print("fold number ", i)
        train_files_path = project_dir + "/data/" + predicate + "/folds_datasets/" + str(i) + "/train/"
        train_files = [os.path.join(train_files_path, f) for f in os.listdir(train_files_path)]
        test_file = project_dir + "/data/" + predicate + "/folds_datasets/" + str(i) + "/test.txt"
        stats = train(not_validate=True, **config)
        stats = validate(**config)
        f_resu.write("%s\t%s\n" % (str(i), repr(stats)))
        f_resu.flush()
os.remove(model_path)