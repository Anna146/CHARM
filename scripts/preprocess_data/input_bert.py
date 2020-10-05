import csv
import random
import numpy as np
import sys
import os
from transformers import DistilBertTokenizer
from transformers.tokenization_distilbert import PRETRAINED_VOCAB_FILES_MAP
from preprocess_utils import *
from collections import defaultdict
import urllib.request
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

def input_bert(predicate = "profession", do_folds = False):
    print("INPUT BERT", "FOLDS" if do_folds else "SEEN", predicate)

    vocab_path = PRETRAINED_VOCAB_FILES_MAP['vocab_file']['distilbert-base-uncased']
    vocab = [line.strip().decode("utf-8") for line in urllib.request.urlopen(vocab_path)]
    ranker_vocab = dict((x[1],x[0]) for x in enumerate(
        [line.strip() for line in open(project_dir + "/data/embeddings/GoogleNews-vectors-negative300_vocab.txt")]))
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
    bert_batch = 4

    predicate_file = project_dir + "/data/" + predicate + "/sources/" + predicate + "_list.txt"
    predicate_list = [line.strip() for line in open(predicate_file, "r")]
    hobby_to_syn = eval(open(project_dir + "/data/" + predicate + "/sources/" + predicate + "_synonyms.txt").read())
    syn_to_hob = dict((syn, val) for val, syns in hobby_to_syn.items() for syn in syns)


    def prepare_files(in_folder, out_folder):
        inp_file = project_dir + "/data/raw_data/texts_" + predicate + ".txt"
        in_test = in_folder + "test.txt"
        in_train = in_folder + "train/"
        test_file = out_folder + "test.txt"
        train_files = out_folder + "train/"

        Path(train_files).mkdir(parents=True, exist_ok=True)

        # Clear contents
        with open(test_file, "w"):
            pass
        for f_name in os.listdir(train_files):
            with open(train_files + f_name, "w"):
                pass

        # Load whitelists
        whitelists = dict()
        for f_name in os.listdir(in_train):
            with open(in_train + f_name, "r") as f_list:
                for uname in f_list:
                    whitelists[uname.strip()] = f_name[:-4]
        with open(in_test, "r") as f_list:
            for uname in f_list:
                whitelists[uname.strip()] = "test"

        num_train = 0
        num_test = 0
        prof_distr = defaultdict(int)

        with open(inp_file, "r") as f_in:
            texts = []
            curr_char = ""
            char_count = 0
            curr_prof_names = set()
            for line in f_in:
                line = line.strip().split(",")
                prof_names = line[1].lower().split(":::")
                if line[0] == curr_char:
                    texts.append(line[2])
                else:
                    if curr_char != "" and curr_char in whitelists:
                        res_file = test_file if whitelists[curr_char] == "test" else train_files + (whitelists[curr_char] + ".txt")
                        examples = []
                        curr_labels = list(set([predicate_list.index(syn_to_hob[x]) for x in set(curr_prof_names)]))
                        examples.append(InputExample(text_a=" ".join(texts), labels=curr_labels, guid=char_count))
                        features = convert_examples_to_features(examples=examples,
                                                                bert_batch=bert_batch, tokenizer=tokenizer,
                                                                vocabulary=vocab, throw_trash=False, ranker_vocab=ranker_vocab)
                        features[0].save(res_file)
                        char_count += 1
                        num_train = num_train + 1 if whitelists[curr_char] != "test" else num_train
                        num_test = num_test + 1 if whitelists[curr_char] == "test" else num_test
                    curr_char = line[0]
                    texts = [line[2]]
                    curr_prof_names = set(prof_names)
                curr_prof_names = curr_prof_names.union(prof_names)


    if not do_folds:
        prepare_files(project_dir + "/data/" + predicate + "/user_whitelists/",
                      project_dir + "/data/" + predicate + "/datasets/")
    else:
        num_folds = 10
        for i in range(num_folds):
            prepare_files(project_dir + "/data/" + predicate + "/folds_whitelists/" + str(i) + "/",
                          project_dir + "/data/" + predicate + "/folds_datasets/" + str(i) + "/")

