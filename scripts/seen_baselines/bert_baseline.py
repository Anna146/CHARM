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
from pytorch_pretrained_bert import BertForSequenceClassification
import random
random.seed(time.time())
from random import random
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent) + "/models")
from compute_statistics import *
from training_utils import *

device = torch.device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
torch.manual_seed(33)
torch.set_printoptions(precision=6, threshold=100000)

parser = argparse.ArgumentParser()
parser.add_argument("--hobby", action="store_true")
parser.add_argument("--profession", action="store_true")

args = parser.parse_args()

##########################  PARAMETERS  ##################################

# Policy params
bert_batch = 4
reduced_input_size = 1

# Optimizer params
batch_size = 2
reg_lambda = 2e-7
bert_lr = 0.00002
fc_lr = 0.001
num_layers = 12

predicate = "profession"
predicate = "profession" if args.profession else "hobby"

#############

# Input files
train_files_path = project_dir + "/data/" + predicate + "/datasets/train/"
train_files = [os.path.join(train_files_path, f) for f in os.listdir(train_files_path)]
test_file = project_dir + "/data/" + predicate + "/datasets/test.txt"
train_file = project_dir + "/data/" + predicate + "/datasets/train.txt"
predicate_path = project_dir + "/data/" + predicate + "/sources/" + predicate + "_list.txt"
predicate_list = [line.strip() for line in open(predicate_path, "r")]
input_weights_path = project_dir + "/data/bert_weigths.npy"
input_vocab_path = PRETRAINED_VOCAB_FILES_MAP['vocab_file']['distilbert-base-uncased']
bert_vocab = [line.strip().decode("utf-8") for line in urllib.request.urlopen(input_vocab_path)]
input_vocab = [line.strip() for line in open(project_dir + "/data/embeddings/GoogleNews-vectors-negative300_vocab.txt")]

shorts = [x[0] for x in enumerate(input_vocab) if len(x[-1]) < 3]
shorts = torch.tensor(shorts).unsqueeze(-1).to(device)

############################################

cls = bert_vocab.index("[CLS]")
sep = bert_vocab.index("[SEP]")

# Training
num_epochs = 100
max_batch_epoch = 500 // batch_size

##################################### MODEL  ##################################

class RLPipeline(nn.Module):
    def __init__(self, **config):
        super(RLPipeline, self).__init__()
        self.policy = BertForSequenceClassification.from_pretrained(project_dir + "/data/embeddings/bert-base-uncased.tar.gz",
                                                                 num_labels=len(predicate_list)).to(device)
        self.policy.bert.encoder.layer = self.policy.bert.encoder.layer[:num_layers]

##################################  Train and eval  ######################


import time
import math

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
    labels = []

    global t1
    print("time to init: ", time.time() - t1)
    curr_segments = []
    curr_words = []
    curr_mask = []

    # Train the Model
    for epoch in range(num_epochs):
        print("Epoch " + str(epoch))
        t1 = time.time()
        for i, train_features in enumerate(mux_stream):
            train_features = train_features[:reduced_input_size]  # reduce the input so that it fits
            for f in train_features:
                f["input_ids"] = f["input_ids"][:-2]
                f["input_ids"] = np.insert(f["input_ids"], cls, 0)
                f["input_ids"] = np.insert(f["input_ids"], sep, len(f["input_ids"]))

            ######## Run policy
            curr_words.append(torch.tensor([f["input_ids"] for f in train_features], dtype=torch.long).to(device))
            curr_mask.append(torch.tensor([f["input_mask"] for f in train_features], dtype=torch.long).to(device))
            curr_segments.append(torch.zeros_like(curr_mask[-1]).to(device))
            label = train_features[0]["label_ids"][0]
            labels.append(label)

            # When the batch is accumulated
            if (i+1) % batch_size == 0:
                optimizer.zero_grad()
                curr_words = torch.cat(curr_words, dim=0).to(device)
                curr_mask = torch.cat(curr_mask, dim=0).to(device)
                curr_segments = torch.cat(curr_segments, dim=0).to(device)
                labels = torch.LongTensor(labels).to(device)

                loss = net.policy(curr_words, curr_segments, curr_mask, labels=labels)
                loss.backward()
                optimizer.step()

                labels = []
                curr_segments = []
                curr_words = []
                curr_mask = []

                if i // batch_size > max_batch_epoch:
                    break

    # save model
    torch.save(net.state_dict(), model_path)

def validate(**config):
    with torch.no_grad():
        # initialize components
        net = RLPipeline(**config)
        net.policy.to(device)
        net.load_state_dict(torch.load(model_path))
        net.eval()

        # input
        streamer = pescador.Streamer(feature_gen, test_file, 4)

        with open(output_file, "w") as f_test_out:
            t1 = time.time()
            ctr = 0
            for train_features in streamer:
                ctr += 1
                train_features = train_features[:reduced_input_size] # reduce the input so that it fits
                for f in train_features:
                    f["input_ids"] = f["input_ids"][:-2]
                    f["input_ids"] = np.insert(f["input_ids"], cls, 0)
                    f["input_ids"] = np.insert(f["input_ids"], sep, len(f["input_ids"]))

                ######## Run policy
                curr_words = torch.tensor([f["input_ids"] for f in train_features], dtype=torch.long).to(device)
                curr_mask = torch.tensor([f["input_mask"] for f in train_features], dtype=torch.long).to(device)
                curr_segments = torch.zeros_like(curr_mask).to(device)

                one_logit = net.policy(curr_words, curr_segments, curr_mask)

                f_test_out.write(str(train_features[0]["guid"]) + "\t" + repr(train_features[0]["label_ids"]) + '\t' + '\t'.join(
                        [str(y) for y in sorted(enumerate(one_logit.cpu().data.numpy()[0]), key=lambda x: x[1], reverse=True)]) + '\n')

            print("test time: ", time.time() - t1)

        stats = compute_whatever_stats(output_file)
        print(stats)
        return stats

##################################################################################

filename = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])
model_path = project_dir + "/data/tmp_folder/" + filename + ".pkl"
seen_dir = project_dir + "/data/" + predicate + "/results/seen_baselines/"
Path(seen_dir).mkdir(parents=True, exist_ok=True)
output_file = seen_dir + "bert.txt"
train()
validate()
os.remove(model_path)