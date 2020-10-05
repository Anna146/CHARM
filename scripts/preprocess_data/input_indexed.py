import os
from nltk import tokenize
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

def input_indexed(predicate = "profession"):
    print("INPUT INDEXED BASELINES", predicate)

    predicate_file = project_dir + "/data/" + predicate + "/sources/" + predicate + "_list.txt"
    predicate_list = [line.strip() for line in open(predicate_file, "r")]
    hobby_to_syn = eval(open(project_dir + "/data/" + predicate + "/sources/" + predicate + "_synonyms.txt").read())
    syn_to_hob = dict((syn, val) for val, syns in hobby_to_syn.items() for syn in syns)

    vocab_path = project_dir + "/data/embeddings/GoogleNews-vectors-negative300_vocab.txt"
    vocab = dict((x[1].strip(), x[0]) for x in enumerate(open(vocab_path).readlines()))
    oov_index = len(vocab)
    max_words = 100
    max_utterance = 100

    def prepare_baseline_files(in_folder, out_folder):
        inp_file = project_dir + "/data/raw_data/texts_" + predicate + ".txt"
        in_test = in_folder + "test.txt"
        in_train = in_folder + "train.txt"
        train_file = out_folder + "indexed_train/indexed_train.txt"
        test_file = out_folder + "indexed_test.txt"

        Path(out_folder + "indexed_train/").mkdir(parents=True, exist_ok=True)

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

        with open(inp_file, "r") as f_in, open(test_file, "a") as f_test, open(train_file, "a") as f_train:
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
                    if curr_char != "" and curr_char in whitelists and curr_char:
                        res_file = f_test if whitelists[curr_char] == "test" else f_train
                        new_texts = []
                        bare_texts = []
                        for text in texts[:max_utterance]:
                            bare_texts.append(text.replace(";", ""))
                            text = [w.lower() for w in tokenize.word_tokenize(text)]
                            text = [vocab[w] if w in vocab else oov_index for w in text][:max_words]
                            text.extend([oov_index] * (max_words - len(text)))
                            assert len(text) == max_words
                            new_texts.append(text)
                        new_texts.extend([[oov_index] * max_words] * (max_utterance - len(texts)))
                        assert len(new_texts) == max_utterance
                        example = [str(x) for t in new_texts for x in t]
                        curr_labels = list(set([predicate_list.index(syn_to_hob[x]) for x in set(curr_prof_names)]))
                        if whitelists[curr_char] != "test":
                            for lab in curr_labels:
                                res_file.write(str(char_count) + ";" + repr([lab]) + ";" + ";".join(example) + ";" + ";".join(bare_texts) + "\n")
                        else:
                            res_file.write(str(char_count) + ";" + repr(curr_labels) + ";" + ";".join(example) + ";" + ";".join(bare_texts) + "\n")
                        char_count += 1
                        num_train = num_train + 1 if whitelists[curr_char] != "test" else num_train
                        num_test = num_test + 1 if whitelists[curr_char] == "test" else num_test
                    curr_char = line[0]
                    texts = [line[2]]
                    curr_prof_names = set(prof_names)
                curr_prof_names = curr_prof_names.union(prof_names)

    prepare_baseline_files(project_dir + "/data/" + predicate + "/user_whitelists/",
                           project_dir + "/data/" + predicate + "/seen_baselines_datasets/")
