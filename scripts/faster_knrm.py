#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 17:20:01 2019

"""
import torch
from torch import nn
import pickle


from pathlib import Path
project_dir = str(Path(__file__).parent.parent)

b0, b1, w0, w1 = pickle.load(open(project_dir + "/data/embeddings/knrm.pkl", "rb"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RbfKernel(nn.Module):
    def __init__(self, initial_mu, initial_sigma, requires_grad=True):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(initial_mu), requires_grad=requires_grad)
        self.sigma = nn.Parameter(torch.tensor(initial_sigma), requires_grad=requires_grad)

    def forward(self, data):
        adj = data - self.mu
        return torch.exp(-0.5 * adj * adj / self.sigma / self.sigma)


class KernelBank(nn.Module):
    def __init__(self, kernels, dim=-1):
        super().__init__()
        self.kernels = nn.ModuleList(kernels)
        self.dim = dim

    def count(self):
        return len(self.kernels)

    def forward(self, data):
        return torch.stack([k(data) for k in self.kernels], dim=self.dim)


class RbfKernelBank(KernelBank):
    def __init__(self, mus=None, sigmas=None, dim=-1, requires_grad=True):
        kernels = [RbfKernel(mu, sigma, requires_grad=requires_grad) for mu, sigma in zip(mus, sigmas)]
        super().__init__(kernels, dim=dim)

    @staticmethod
    def from_strs(
        mus="-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.0",
        sigmas="0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.001",
        dim=-1,
        requires_grad=True,
    ):
        mus = [float(x) for x in mus.split(",")]
        sigmas = [float(x) for x in sigmas.split(",")]
        return RbfKernelBank(mus, sigmas, dim=dim, requires_grad=requires_grad)

    @staticmethod
    def evenly_spaced(count=11, sigma=0.1, rng=(-1, 1), dim=-1, requires_grad=True):
        mus = [x.item() for x in torch.linspace(rng[0], rng[1], steps=count)]
        sigmas = [sigma for _ in mus]
        return RbfKernelBank(mus, sigmas, dim=dim, requires_grad=requires_grad)


import numpy as np

few_kernels_flag = True

class faster_KNRM(nn.Module):
    #    @staticmethod
    @staticmethod
    def default_config():
        # passagelen = 6
        # self.p["maxqlen"], EMBEDDING_DIM, BATCH_SIZE come from main config

        gradkernels = False
        singlefc = False#True
        kernel = "rbf"
        return locals().copy()

    def __init__(self, weights_matrix, p):
        super(faster_KNRM, self).__init__()
        self.p = p
        self.size = weights_matrix.shape[1]
        weights_matrix = np.concatenate([np.zeros((1, self.size)), weights_matrix])
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(weights_matrix.astype(np.float32)))
        if few_kernels_flag:
            mus = '-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.0'#"0.5,0.7,0.9,1.0"
            sigmas = '0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.001'#"0.1,0.1,0.1,0.001"
        else:
            mus = "1.0,1.0"
            sigmas = "0.0,0.001"
        self.kernels = RbfKernelBank.from_strs(mus, sigmas, dim=1, requires_grad=p["gradkernels"])

        if p["singlefc"]:
            self.combine = nn.Linear(self.kernels.count() * 1, 1)
        else:
            self.combine1 = nn.Linear(self.kernels.count() * 1, 30)
            self.combine2 = nn.Linear(30, 1)
            self.combine1.weight.data = torch.nn.Parameter(torch.tensor(w0).t())
            self.combine1.bias.data = torch.nn.Parameter(torch.tensor(b0))
            self.combine2.weight.data = torch.nn.Parameter(torch.tensor(w1).t())
            self.combine2.bias.data = torch.nn.Parameter(torch.tensor(b1))

        self.padding = -1


    def input_spec(self):
        result = super().input_spec()
        result["fields"].update({"query_tok", "doc_tok", "query_len", "doc_len"})
        # combination does not enforce strict lengths for doc or query
        result["qlen_mode"] = "max"
        result["dlen_mode"] = "max"
        return result

    def forward(self, sentence, query_sentence):
        batch_size = sentence.size()[0]
        qlen = query_sentence.size()[1]
        BAT, A, B = query_sentence.shape[0], query_sentence.shape[1], sentence.shape[1]
        query_sentence = query_sentence.to(device)
        sentence = sentence.to(device)
        x = self.embedding(sentence + 1).to(device)
        query_x = self.embedding(query_sentence + 1).to(device)
        x_norm = x
        query_x_norm = query_x
        M_cos = torch.matmul(query_x_norm, torch.transpose(x_norm, 1, 2))
        nul = torch.zeros_like(M_cos)
        simmat = torch.where(query_sentence.reshape(BAT, A, 1).expand(BAT, A, B) == self.padding, nul, M_cos)
        simmat = torch.where(sentence.reshape(BAT, 1, B).expand(BAT, A, B) == self.padding, nul, simmat)
        simmat = simmat.reshape(simmat.shape[0], 1, simmat.shape[1], simmat.shape[2])
        kernels = self.kernels(simmat)
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        simmat = (
            simmat.reshape(BATCH, 1, VIEWS, QLEN, DLEN)
                .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN)
                .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        )
        result = kernels.sum(dim=3)  # sum over document
        mask = simmat.sum(dim=3) != 0.0  # which query terms are not padding?
        result = torch.where(mask, (result + 1e-6).log(), mask.float())
        result = result.sum(dim=2)  # sum over query terms


        if self.p["singlefc"]:
            scores = self.combine(result)  # linear combination over kernels
        else:
            scores1 = torch.tanh(self.combine1(result))
            scores = torch.tanh(self.combine2(scores1))
        return scores

    def path_segment(self):
        result = "{base}_{kernel}".format(base=super().path_segment(), **self.config)
        if not self.config["gradkernels"]:
            result += "_nogradkernels"
        return result

    def no_save(self):
        return self.simmat.embedding.no_save("simmat.embedding")

    def val_per_mu(self, sentence, query_sentence, sentence_sizes):

        batch_size = sentence.size()[0]
        qlen = query_sentence.size()[1]
        query_sentence = query_sentence.to(device)
        sentence = sentence.to(device)
        x = self.embedding(sentence + 1).to(device)
        query_x = self.embedding(query_sentence + 1).to(device)
        x_norm = x
        query_x_norm = query_x
        M_cos = torch.matmul(query_x_norm, torch.transpose(x_norm, 1, 2))
        simmat = M_cos.reshape(batch_size, 1, qlen, -1)
        kernels = self.kernels(simmat)
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        simmat = (
            simmat.reshape(BATCH, 1, VIEWS, QLEN, DLEN)
            .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN)
            .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        )
        result = kernels.sum(dim=3)  # sum over document
        mask = simmat.sum(dim=3) != 0.0  # which query terms are not padding?
        result = torch.where(mask, (result + 1e-6).log(), mask.float())
        return result
