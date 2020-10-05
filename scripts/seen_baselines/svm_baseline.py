from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
import sys
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

sys.path.insert(0, str(Path(__file__).parent.parent))
from compute_statistics import *
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hobby", action="store_true")
parser.add_argument("--profession", action="store_true")

args = parser.parse_args()

pred = "profession"
pred = "profession" if args.profession else "hobby"

def svm_baseline(predicate):

    in_train = project_dir + "/data/" + predicate + "/seen_baselines_datasets/train.txt"
    in_test = project_dir + "/data/" + predicate + "/seen_baselines_datasets/test.txt"

    pred_num = len(open(project_dir + "/data/" + predicate + "/sources/" + predicate + "_list.txt").readlines())

    res_file = project_dir + "/data/" + predicate + "/results/seen_baselines/svm.txt"
    Path(project_dir + "/data/" + predicate + "/results/seen_baselines/").mkdir(parents=True, exist_ok=True)

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    test_index = []

    with open(in_train, "r") as f_in:
        reader = csv.reader(f_in)
        for line in reader:
            vals = [int(x) for x in line[0].split(" ")]
            for val in vals:
                y_train.append(val)
                X_train.append(line[1])

    with open(in_test, "r") as f_in:
        cnt = 0
        reader = csv.reader(f_in)
        for line in reader:
            vals = [int(x) for x in line[0].split(" ")]
            for val in vals:
                y_test.append(vals)
                X_test.append(line[1])
                test_index.append(cnt)
            cnt += 1

    vectorizer1 = TfidfVectorizer(decode_error="ignore", lowercase=True, analyzer="word", ngram_range=(1, 2),
                                 min_df=2, max_features=None, use_idf=True, smooth_idf=True, sublinear_tf=True)
    vectorizer2 = TfidfVectorizer(decode_error="ignore", lowercase=True, analyzer="char", ngram_range=(3, 5),
                                 min_df=2, max_features=None, use_idf=True, smooth_idf=True, sublinear_tf=True)

    vectorizer1.fit(X_test+ X_train)
    vectorizer2.fit(X_test+ X_train)

    X1 = vectorizer2.transform(X_train)
    X2 = vectorizer2.transform(X_train)

    X_train = hstack([X1, X2])

    X1 = vectorizer2.transform(X_test)
    X2 = vectorizer2.transform(X_test)
    X_test = hstack([X1, X2])

    print("started classifier")
    clf = LinearSVC(max_iter=100, C=0.001)
    clf.fit(X_train, y_train)
    print(clf.classes_)

    all_classes = np.array(sorted([x for x in range(pred_num)]))
    # Get the probabilities for learnt classes
    prob_pos = clf.decision_function(X_test)
    prob = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    # Create the result matrix, where all values are initially zero
    probas = np.zeros((prob.shape[0], all_classes.size))
    # Set the columns corresponding to clf.classes_
    probas[:, all_classes.searchsorted(clf.classes_)] = prob

    already_was = set()

    with open(res_file, "w") as f_out:
        ch_id = 0
        for i in range(len(probas)):
            if test_index[i] in already_was:
                continue
            already_was.add(test_index[i])
            pr = sorted(enumerate(probas[i]), key=lambda x:x[1], reverse=True)
            f_out.write(str(ch_id) + "\t" + str(y_test[i]) + "\t" + "\t".join(str(y) for y in pr) + "\n")
            ch_id += 1

    stats = compute_whatever_stats(res_file)
    print(stats)

svm_baseline(pred)
