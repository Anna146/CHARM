import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from pathlib import Path
project_dir = str(Path(__file__).parent.parent)

#model_file = project_dir + "/data/embeddings/GoogleNews-vectors-negative300.txt"
#model_file = "/home/tigunova/GoogleNews-vectors-negative300-SLIM.txt"
model = KeyedVectors.load_word2vec_format(project_dir + '/data/embeddings/GoogleNews-vectors-negative300.bin', binary=True)
#model = KeyedVectors.load_word2vec_format("/home/tigunova/Documents/google-models/GoogleNews-vectors-negative300-SLIM.bin", binary=True)

matrix = []
with open(project_dir + "/data/embeddings/GoogleNews-vectors-negative300_vocab.txt", "w") as f_w:
    for k in model.wv.vocab:
        f_w.write(k + "\n")
        matrix.append(model.wv[k])
np.save(project_dir + "/data/embeddings/GoogleNews-vectors-negative300.npy", np.array(matrix))
print("saved")