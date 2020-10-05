import os
import pytextrank
import json
import spacy
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

predicates = ["hobby", "profession"]

### Prepare json
nlp = spacy.load("en_core_web_sm")
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
max_keyw = 50

num_folds = 10
for predicate in predicates:
    for i in range(num_folds):
        print("fold", i)
        inp = project_dir + "/data/" + predicate + "/seen_baselines_datasets/folds/%d/test.txt" % (i)
        output_dir = project_dir + "/data/" + predicate + "/queries/textrank/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cnt = 0
        with open(inp) as f_in, open(output_dir + str(i) + ".txt", "w") as ff:
            for line in f_in:
                line = line.strip().split(",")
                doc = nlp(line[1])
                all_kw = []
                for p in doc._.phrases:
                    all_kw.append(p.text)
                    if len(all_kw) >= max_keyw:
                        break
                ff.write(",".join([line[0]] + [" ".join(all_kw)]) + "\n")