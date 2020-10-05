from collections import Counter, defaultdict
from pprint import pprint
import random
import os
import numpy as np

from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

def make_whitelists(predicate = "profession"):
    inp_dict = eval(open(project_dir + "/data/" + predicate + "/splits_" + predicate + "_dictionary.txt").read())
    seen_dir = project_dir + "/data/" + predicate + "/user_whitelists/"
    Path(seen_dir).mkdir(parents=True, exist_ok=True)
    for tname in ["train", "test"]:
        open(seen_dir + tname + ".txt", "w").write("\n".join(inp_dict["seen"][tname]))
        for i in range(10):
            fold_dir = project_dir + "/data/" + predicate + "/folds_whitelists/" + str(i) + "/"
            Path(fold_dir).mkdir(parents=True, exist_ok=True)
            open(fold_dir + tname + ".txt", "w").write("\n".join(inp_dict[i][tname]))


    train_file = project_dir + "/data/" + predicate + "/user_whitelists/train.txt"
    train_dir = str(project_dir + "/data/" + predicate + "/user_whitelists/train/")

    Path(train_dir).mkdir(parents=True, exist_ok=True)
    train_guys = [line.strip() for line in open(train_file)]

    num_files = 30
    per_file = len(train_guys) // num_files + 1
    for i in range(num_files):
        with open(train_dir + str(i) + ".txt", "w") as f_train:
            f_train.write("\n".join(train_guys[i * per_file: min((i + 1) * per_file, len(train_guys))]))

    num_files = 30
    for fold in range(10):
        folds_dir = str(project_dir + "/data/" + predicate + "/folds_whitelists/")
        folds_file = str(folds_dir + str(fold) + "/train.txt")
        train_guys = [line.strip() for line in open(folds_file)]
        per_file = len(train_guys) // num_files + 1
        for i in range(num_files):
            Path(os.path.join(folds_dir, str(fold), "train")).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(folds_dir, str(fold), "train") + "/" + str(i) + ".txt", "w") as f_train:
                f_train.write("\n".join(train_guys[i * per_file: min((i + 1) * per_file, len(train_guys))]))

