import re
import csv
import os
from preprocess_utils import *
from transformers import DistilBertTokenizer
from pathlib import Path

project_dir = str(Path(__file__).parent.parent.parent)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
ranker_vocab = dict((x[1],x[0]) for x in enumerate(
    [line.strip() for line in open(project_dir + "/data/embeddings/GoogleNews-vectors-negative300_vocab.txt")]))

def docs_for_doublebert(predicate = "profession", collection="shr"):
    print("finished documents for bert IR", predicate, collection)

    predicate_path = project_dir + "/data/" + predicate + "/sources/" + predicate + "_list.txt"
    predicate_list = [line.strip() for line in open(predicate_path, "r")]

    name = "_".join([predicate, collection])

    inp_docs = project_dir + "/data/documents/" + name + ".txt"
    out_docs = project_dir + "/data/double_bert_documents/" + name + ".txt"

    Path(project_dir + "/data/double_bert_documents/").mkdir(parents=True, exist_ok=True)

    open(out_docs, "w")
    from collections import defaultdict
    out_counts = defaultdict(int)

    goods = 0
    bads = 0
    examples = []
    with open(inp_docs, "r") as f_in:
        for line1 in f_in:
            line = line1.strip().replace("\n", " ").split("\t")
            if len(line) < 2:
                bads += 1
                continue

            article_name = line[-1][:100]
            label = line[0].lower()
            if label not in predicate_list:
                continue
            out_counts[predicate_list.index(label)] += 1
            examples.append(InputExample(text_a=" ".join([line[-1], " ", line[1]]),
                                         labels=[predicate_list.index(label)], guid=goods))
            doc_features = convert_examples_to_features(examples=examples, bert_batch=2, tokenizer=tokenizer,
                                                        ranker_vocab=ranker_vocab, plain_texts=article_name)
            examples = []
            goods += 1
            doc_features[0].save(out_docs)
