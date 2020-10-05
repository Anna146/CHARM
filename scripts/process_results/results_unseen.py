import sys
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

sys.path.insert(0, str(Path(__file__).parent.parent))
from compute_statistics import *
from scipy.stats import ttest_rel

num_folds = 10

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hobby", action="store_true")
parser.add_argument("--profession", action="store_true")

args = parser.parse_args()

predicate = "profession"
predicate = "profession" if args.profession else "hobby"

collection = "shr"
methods = ["bert_bm25", "bert_knrm", "bert_ir", "fulltext", "rake", "textrank"]

metrics_dict = dict()
dicts_dict = dict()

for method in methods:
    try:
        predicate_path = project_dir + "/data/" + predicate + "/sources/" + predicate + "_list.txt"
        predicate_list = [line.strip() for line in open(predicate_path, "r")]
        counts_dict = eval(open(project_dir + "/data/%s_counts.txt" % predicate).read())

        inp_files = [project_dir + "/data/%s/results/%s/output_files/folds/%s/%d.txt" % (predicate, collection, method, i) for i in range(num_folds)]
        out_file = project_dir + "/data/%s/results/%s/output_files/folds/%s.txt" % (predicate, collection, method)
        with open(out_file, 'w') as outfile:
            for fname in inp_files:
                with open(fname) as infile:
                    outfile.write(infile.read())
        metrics_dict[method], dicts_dict[method] = compute_whatever_stats(out_file, with_dict=True)
        metrics = metrics_dict[method].keys()
    except Exception as e:
        print(e)
        continue

print(metrics_dict)