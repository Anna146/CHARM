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
import math
import os
import argparse
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent) + "/models")
from compute_statistics import *
from _bm25 import *
from training_utils import *

torch.manual_seed(33)
torch.set_printoptions(precision=6, threshold=100000)

parser = argparse.ArgumentParser()
parser.add_argument("--shr", action="store_true")
parser.add_argument("--ext", action="store_true")
parser.add_argument("--pat", action="store_true")
parser.add_argument("--hobby", action="store_true")
parser.add_argument("--profession", action="store_true")

args = parser.parse_args()

##########################  PARAMETERS  ##################################

predicate = "hobby"
predicate = "profession" if args.profession else "hobby"

# Input files
predicate_path = project_dir + "/data/" + predicate + "/sources/" + predicate + "_list.txt"
predicate_list = [line.strip() for line in open(predicate_path, "r")]
input_vocab = [line.strip() for line in open(project_dir + "/data/embeddings/GoogleNews-vectors-negative300_vocab.txt")]
inverse_vocab = dict((x[1], x[0]) for x in enumerate(input_vocab))

############################################

# IR params
exp_name = "shr"
exp_name = "ext" if args.ext else exp_name
exp_name = "pat" if args.pat else exp_name

collection_path = project_dir + "/data/bert_documents/" + predicate + "_" + exp_name + ".txt"
collection_tfs = project_dir + "/data/bert_documents/" + predicate + "_" + exp_name + "_tfs.txt"

doc_len = 800
agg_type = "max"
k1 = 2.0
b = 0.75

##################################### MODEL  ##################################

doc_tensors, doc_labels, real_docs, df_dict, doc_dict, tfs_dict, doc_ids = load_documents(collection_path, predicate_list, doc_len=doc_len, tfs_file = collection_tfs)
ranker = bm25(df_dict=df_dict, k1=k1, b=b, doc_num=len(doc_tensors), doc_len=doc_len)
curr_docs = torch.stack(doc_tensors).squeeze()
curr_labels = torch.LongTensor(doc_labels)


def validate(input_file, output_file):

    with open(output_file, "w") as f_test_out:
        ctr = 0
        for line in open(input_file):
            line = line.strip().split(",")
            label = line[0].split(" ")
            query = [inverse_vocab.get(term, 0) for term in line[1].split(" ")]
            scores_list = []
            df = [ranker.df_dict.get(term, 0) for term in query]
            idf = torch.tensor([math.log((ranker.doc_num - termdf + 0.5) / (termdf + 0.5)) for termdf in df])
            for num, doc in zip(doc_ids, curr_docs):
                qtf = torch.FloatTensor([tfs_dict[num].get(term, 0) for term in query])
                scores_list.append(ranker(qtf, idf, doc))
            scores_list = torch.stack(scores_list)
            scores_list = aggregate_scores(scores_list, curr_labels, len(predicate_list), agg_type)
            scores_list = torch.where(scores_list != 0, scores_list, torch.ones_like(scores_list) * np.NINF)
            scores_list = scores_list.cpu().data.numpy()

            f_test_out.write(str(ctr) + "\t" + repr([int(y) for y in label]) + '\t' + '\t'.join(
                    [str(y) for y in sorted(enumerate(scores_list), key=lambda x: x[1], reverse=True)]) + '\n')
            ctr += 1
    stats = compute_whatever_stats(output_file)
    print(stats)
    return stats

##########################################

####################################  MAIN  #####################################################################
methods = ["fulltext", "textrank", "rake"]

num_folds = 10

for method in methods:
    input_dir = project_dir + "/data/" + predicate + "/queries/" + method + "/"
    output_dir = project_dir + "/data/" + predicate + "/results/" + exp_name + "/output_files/folds/" + method + "/"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(project_dir + "/data/" + predicate + "/results/" + exp_name + "/folds_" + method + ".txt", "w") as f:
        for i in range(num_folds):
            if method == "fulltext":
                stats = validate(input_file=project_dir + "/data/" + predicate + "/seen_baselines_datasets/folds/" + str(i) + "/test.txt", output_file=output_dir + str(i) + ".txt")
            else:
                stats = validate(input_file=input_dir + str(i) + ".txt", output_file=output_dir + str(i) + ".txt")
            f.write(str(i) + "\t" + repr(stats) + "\n")
            f.flush()


