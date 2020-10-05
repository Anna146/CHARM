from rake_nltk import Rake
import os
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

predicates = ["profession", "hobby"]

r = Rake(max_length=3) # Uses stopwords for english from NLTK, and all puntuation characters.
max_keyw = 50

num_folds = 10

for predicate in predicates:
    for i in range(num_folds):
        inp = project_dir + "/data/" + predicate + "/seen_baselines_datasets/folds/%d/test.txt" % (i)
        output_dir = project_dir + "/data/" + predicate + "/queries/rake/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(inp) as f_in, open(output_dir + str(i) + ".txt", "w") as ff:
            for line in f_in:
                line = line.strip().split(",")
                r.extract_keywords_from_text(line[1])
                phrases = r.get_ranked_phrases()[:max_keyw]
                if len(phrases) < max_keyw:
                    phrases.extend(["-1"] * (max_keyw - len(phrases)))
                ff.write(",".join([line[0]] + [" ".join(phrases[:max_keyw])]) + "\n")
            print("done fold", i)
