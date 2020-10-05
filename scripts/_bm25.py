import torch
from torch import nn
import math



class bm25(nn.Module):
    def __init__(self, df_dict, k1, b, doc_num=0, doc_len=0):
        super(bm25, self).__init__()
        self.k1 = torch.tensor(k1)
        self.b = torch.tensor(b)
        self.doc_num, self.doc_len = doc_num, doc_len
        self.df_dict = {tok: torch.tensor(df) for tok, df in df_dict.items()} if df_dict != None else None

    def forward(self, qtf, idf, d, without_idf = False):
        num = qtf * (self.k1 + 1)
        denom = qtf + self.k1 * (1 - self.b + self.b * 3 / 3)  # TODO fix hardcoded 3

        if not without_idf:
            scores = idf * (num / denom)
        else:
            scores = num /denom

        return torch.sum(scores)