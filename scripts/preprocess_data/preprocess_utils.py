import numpy as np
import pickle
from numpy.linalg import norm
from nltk.corpus import stopwords
from pathlib import Path

import pickle

class InputFeatures(object):
    def __init__(self, input_ids = None, input_mask = None, segment_ids = None, label_ids = None, guid = None,
                 hash_mask = None, ranker_tokens = None, plain_texts = None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.hash_mask = hash_mask
        self.label_ids = label_ids
        self.guid = guid
        self.ranker_tokens = ranker_tokens
        self.plain_texts = plain_texts

    def save(self, file):
        with open(file, "ab") as f:
            pickle.dump(self.__dict__, f)

    def load(self, f):
        self.__dict__ = pickle.load(f)


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, labels=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels

stop_words = set(stopwords.words('english'))

def feature_gen(filepath, batch_size):
    with open(filepath, "rb") as f_in:
        while True:
            batch = []
            sample = InputFeatures()
            try:
                sample.load(f_in)
            except (EOFError, pickle.UnpicklingError):
                raise StopIteration
            for inp_id, inp_msk, seg_id, hash_msk in zip(np.split(np.array(sample.input_ids), batch_size),
                                                         np.split(np.array(sample.input_mask), batch_size),
                                                         np.split(np.array(sample.segment_ids), batch_size),
                                                         np.split(np.array(sample.hash_mask), batch_size)):
                batch.append(dict(input_ids=inp_id,
                              input_mask=inp_msk,
                              segment_ids=seg_id,
                              label_id=sample.label_id,
                              guid=sample.guid,
                              hash_mask=hash_msk
                            ))
            yield batch

project_dir = str(Path(__file__).parent.parent.parent)
stopwords = set(line.strip() for line in open(project_dir + "/scripts/stoplist.txt", "r"))

from nltk import word_tokenize

def docs_to_indexed(text, ranker_vocab = None, throw_trash = True):
    tokens = word_tokenize(text)
    if throw_trash:
        tokens = [x for x in tokens if x not in stopwords and len(x) > 2 and x.isalpha()]
    return [ranker_vocab[x] for x in tokens if x in ranker_vocab]

def convert_examples_to_features(examples, bert_batch, tokenizer, vocabulary = None, pad = True, throw_trash = False,
                                 ranker_vocab = None, plain_texts = None):
    max_seq_length = 512 * bert_batch

    features = []
    example = examples[0]

    tokens_a = tokenizer.tokenize(example.text_a)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length:# - 2:
        tokens_a = tokens_a[0:(max_seq_length)]# - 2)]

    tokens = []
    segment_ids = []
    ranker_tokens = []
    for i in range(len(tokens_a)):
        curr_tok = tokens_a[i]
        tokens.append(curr_tok)
        if curr_tok[0] == "#":
            ranker_tokens.append(0)
            continue
        postfixes = [curr_tok]
        j = i+1
        while j < len(tokens_a) and tokens_a[j][0] == "#":
            postfixes.append(tokens_a[j].replace("#", ""))
            j += 1
        try_word = 0

        while len(postfixes) > 0:
            try_word = ranker_vocab.get("".join(postfixes), 0)
            if try_word != 0:
                break
            postfixes = postfixes[:-1]
        ranker_tokens.append(try_word)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    assert len(tokens) == len(ranker_tokens) == len(input_ids)
    input_mask = [1] * len(input_ids)
    if vocabulary != None:
        hash_mask = [0 if any([y in id for y in ["[", "#"]]) else 1 for id in tokens]
    else:
        hash_mask = input_mask
    hash_mask = [hash_mask[k] if tokens[k] not in stop_words and ranker_tokens[k] != 0 else 0 for k in range(len(hash_mask))]

    # Zero-pad up to the sequence length.
    if pad == True:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            hash_mask.append(0)
            ranker_tokens.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

    label_ids = example.labels

    features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids,
                          guid=example.guid,
                          hash_mask=hash_mask,
                          ranker_tokens=ranker_tokens,
                          plain_texts=plain_texts))
    return features

from collections import Counter
def get_lens():
    print(sorted(dict(Counter(big_lens)).items(), key=lambda x: x[0]))
    
def load_weights_normalized(path):
    input_weights = np.load(path).astype(float)
    nor = np.expand_dims(norm(input_weights, axis=1), 1)
    input_weights /= (nor + 1e-7)
    vocab_len = input_weights.shape[0]
    input_weights = np.append(input_weights, np.zeros(shape=(1, input_weights.shape[1])), axis=0)
    return input_weights, vocab_len