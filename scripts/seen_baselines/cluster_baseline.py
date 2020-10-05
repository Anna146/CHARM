import os
import csv
import sys
from sklearn.gaussian_process import GaussianProcessClassifier
import numpy
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

sys.path.insert(0, str(Path(__file__).parent.parent))
from compute_statistics import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hobby", action="store_true")
parser.add_argument("--profession", action="store_true")

args = parser.parse_args()

pred = "profession"
pred = "profession" if args.profession else "hobby"

def cluster_baseline(predicate):
    num_clusters = 100

    inp_train = project_dir + "/data/" + predicate + "/seen_baselines_datasets/train.txt"
    inp_test = project_dir + "/data/" + predicate + "/seen_baselines_datasets/test.txt"
    pred_num = len(open(project_dir + "/data/" + predicate + "/sources/" + predicate + "_list.txt").readlines())
    res_file = project_dir + "/data/" + predicate + "/results/seen_baselines/clusters.txt"
    Path(project_dir + "/data/" + predicate + "/results/seen_baselines/").mkdir(parents=True, exist_ok=True)

    cluster_path = project_dir + "/data/embeddings/glove-100"
    vocab = dict((x.strip().split(" ")[0], int(x.strip().split(" ")[2]) - 1) for x in open(cluster_path, "r"))

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    test_index = []

    with open(inp_train, "r") as f_in:
        for line in f_in:
            line = line.strip().split(",")
            vals = [int(x) for x in line[0].split(" ")]
            features = [0 for x in range(num_clusters)]
            all_words = 0
            line = line[-1].split(" ")
            for w in line:
                if w in vocab:
                    features[vocab[w]] += 1
                    all_words += 1
            features = [x * 1.0 / all_words for x in features]
            for val in vals:
                y_train.append(val)
                X_train.append(features)

    with open(inp_test, "r") as f_in:
        cnt = 0
        for line in f_in:
            line = line.strip().split(",")
            vals = [int(x) for x in line[0].split(" ")]
            features = [0 for x in range(num_clusters)]
            all_words = 0
            line = line[-1].split(" ")
            for w in line:
                if w in vocab:
                    features[vocab[w]] += 1
                    all_words += 1
            features = [x * 1.0 / all_words for x in features]
            for val in vals:
                X_test.append(features)
                test_index.append(cnt)
                y_test.append(vals)
            cnt += 1

    classifier = GaussianProcessClassifier(kernel=None, max_iter_predict=100, warm_start=False, copy_X_train=False,
                                           random_state=33, multi_class= "one_vs_rest").fit(X_train, y_train)

    all_classes = numpy.array(sorted([x for x in range(pred_num)]))
    # Get the probabilities for learnt classes
    prob = classifier.predict_proba(X_test)
    # Create the result matrix, where all values are initially zero
    probas = numpy.zeros((prob.shape[0], all_classes.size))
    # Set the columns corresponding to clf.classes_
    probas[:, all_classes.searchsorted(classifier.classes_)] = prob

    already_was = set()
    with open(res_file, "w") as f_out:
        ch_id = 0
        for i in range(len(probas)):
            if test_index[i] in already_was:
                continue
            already_was.add(test_index[i])
            pr = sorted(enumerate(probas[i]), key=lambda x:x[1], reverse=True)
            f_out.write(str(ch_id) + "\t" + repr(y_test[i]) + "\t" + "\t".join(str(y) for y in pr) + "\n")
            ch_id += 1

    stats = compute_whatever_stats(res_file)
    print(stats)

cluster_baseline(pred)