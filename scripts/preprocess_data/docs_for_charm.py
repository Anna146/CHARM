import re
import os
import csv
from preprocess_utils import *
from transformers import DistilBertTokenizer
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

ranker_vocab = dict((x[1],x[0]) for x in enumerate(
    [line.strip() for line in open(project_dir + "/data/embeddings/GoogleNews-vectors-negative300_vocab.txt")]))


def docs_for_charm(predicate = "profession", collection = "shr"):
    print("creating documents for charm", predicate, collection)
    name = "_".join([predicate, collection])

    inp_docs = project_dir + "/data/documents/" + name + ".txt"
    out_docs = project_dir + "/data/bert_documents/" + name + ".txt"
    out_tfs = project_dir + "/data/bert_documents/" + name + "_tfs.txt"

    Path(project_dir + "/data/bert_documents/").mkdir(parents=True, exist_ok=True)

    limit = 6000

    from collections import Counter
    import pickle

    tf_dict = dict()

    goods = 0
    bads = 0
    with open(inp_docs, "r") as f_in, open(out_docs, "w") as f_out, open(out_tfs, "wb") as f_tfs:
        writer = csv.writer(f_out)
        for line1 in f_in:
            line = line1.strip().replace("\n", " ").split("\t")
            if len(line) < 2:
                bads += 1
                continue

            article_name = line[-1][:100]
            label = line[0]
            doc = docs_to_indexed(line[1], ranker_vocab = ranker_vocab, throw_trash = True)
            if len(doc) > 5:
                writer.writerow([label] + doc[:limit] + [article_name.replace(":", "") + ":::" + str(goods)])  # [line[1].replace("\n", "").replace(",", "").replace("\t","")[:500]])
                tf_dict[goods] = dict(Counter(doc))
                goods += 1
            else:
                bads += 1
        pickle.dump(tf_dict, f_tfs)

docs_for_charm("hobby")
