import os
from nltk import tokenize
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

def input_plaintext(predicate = "profession", do_folds = False):
    print("INPUT PLAINTEXT", "FOLDS" if do_folds else "SEEN", predicate)

    predicate_file = project_dir + "/data/" + predicate + "/sources/" + predicate + "_list.txt"
    predicate_list = [line.strip() for line in open(predicate_file, "r")]
    hobby_to_syn = eval(open(project_dir + "/data/" + predicate + "/sources/" + predicate + "_synonyms.txt").read())
    syn_to_hob = dict((syn, val) for val, syns in hobby_to_syn.items() for syn in syns)

    def prepare_baseline_files(in_folder, out_folder):
        inp_file = project_dir + "/data/raw_data/texts_" + predicate + ".txt"
        in_test = in_folder + "test.txt"
        in_train = in_folder + "train.txt"
        train_file = out_folder + "train.txt"
        test_file = out_folder + "test.txt"

        Path(out_folder).mkdir(parents=True, exist_ok=True)

        # Clear contents
        with open(test_file, "w"), open(train_file, "w"):
            pass

        # Load whitelists
        whitelists = dict()
        with open(in_test, "r") as f_list:
            for uname in f_list:
                whitelists[uname.strip()] = "test"
        with open(in_train, "r") as f_list:
            for uname in f_list:
                whitelists[uname.strip()] = "train"

        num_train = 0
        num_test = 0
        seen_guys = set()
        from collections import defaultdict
        prof_distr = defaultdict(int)
        count_pairs = 0

        with open(inp_file, "r") as f_in, open(test_file, "a") as f_test, open(train_file, "a") as f_train:
            texts = []
            curr_char = ""
            char_count = 0
            curr_prof_names = set()
            for line in f_in:
                line = line.strip().split(",")
                seen_guys.add(line[0])
                prof_names = line[1].lower().split(":::")
                if line[0] == curr_char:
                    texts.append(line[2])
                else:
                    if curr_char != "" and curr_char in whitelists:
                        res_file = f_test if whitelists[curr_char] == "test" else f_train
                        example = [w.lower() for w in tokenize.word_tokenize(" ".join(texts))]
                        curr_labels = list(set([predicate_list.index(syn_to_hob[x]) for x in set(curr_prof_names)]))
                        if len(curr_labels) > 1:
                            count_pairs += 1
                        res_file.write(" ".join(str(x) for x in curr_labels) + "," + " ".join(example) + "\n")
                        for x in curr_labels:
                            prof_distr[x] += 1
                        char_count += 1
                        num_train = num_train + 1 if whitelists[curr_char] != "test" else num_train
                        num_test = num_test + 1 if whitelists[curr_char] == "test" else num_test
                    curr_char = line[0]
                    texts = [line[2]]
                    curr_prof_names = set(prof_names)
                curr_prof_names = curr_prof_names.union(prof_names)

    if not do_folds:
        prepare_baseline_files(project_dir + "/data/" + predicate + "/user_whitelists/",
                               project_dir + "/data/" + predicate + "/seen_baselines_datasets/")
    else:
        num_folds = 10
        for i in range(num_folds):
            prepare_baseline_files(project_dir + "/data/" + predicate + "/folds_whitelists/" + str(i) + "/",
                                   project_dir + "/data/" + predicate + "/seen_baselines_datasets/folds/" + str(i) + "/")


