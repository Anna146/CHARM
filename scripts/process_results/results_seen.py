import sys
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

sys.path.insert(0, str(Path(__file__).parent.parent))
from compute_statistics import *
import os
from scipy.stats import ttest_rel

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hobby", action="store_true")
parser.add_argument("--profession", action="store_true")

args = parser.parse_args()

predicate = "profession"
predicate = "profession" if args.profession else "hobby"


collects = ["shr", "ext", "pat"]
methods = ["bert_bm25", "bert_knrm"]
all_methods = []

metrics_dict = dict()
dicts_dict = dict()

baseline_path = project_dir + "/data/" + predicate + "/results/seen_baselines/"
baselines = ["clusters", "svm", "bert"]

for method in baselines:
    metrics_dict[method], dicts_dict[method] = compute_whatever_stats(os.path.join(baseline_path, method + ".txt"), with_dict=True)
    all_methods.append(method)
print("loaded baselines")

for collection in collects:
    input_folder = project_dir + "/data/%s/results/%s/output_files/seen/" % (predicate, collection)
    predicate_path = project_dir + "/data/" + predicate + "/sources/" + predicate + "_list.txt"
    predicate_list = [line.strip() for line in open(predicate_path, "r")]
    counts_dict = eval(open(project_dir + "/data/%s_counts.txt" % predicate).read())
    all_stats = []

    for fi in os.listdir(input_folder):
        stats = compute_whatever_stats(os.path.join(input_folder, fi))
        metrics_dict[fi.strip(".txt") + "_" + collection], dicts_dict[fi.strip(".txt") + "_" + collection] = compute_whatever_stats(
            os.path.join(input_folder, fi), with_dict=True)
        
print(metrics_dict)