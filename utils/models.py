"""
Implementation of simple, explainable models and nn loaders
"""

import os

import torch
from torch import nn

import numpy as np
import scipy


class Product:
    def __init__(self, vectorizer, w_list):
        # returns 1 if words in w_list are present in the example
        self.vectorizer = vectorizer
        self.w_list = w_list
        self.ids = [vectorizer.vocabulary_[w] for w in w_list]

    # CLASSIFIER
    def predict_proba(self, docs):
        outs = np.zeros((len(docs), 2))
        vect = self.vectorizer.transform(docs)
        for i, x in enumerate(docs):
            if all([vect[i, j] > 0 for j in self.ids]):
                outs[i, 1] = 1
        outs[:, 0] = 1 - outs[:, 1]
        return outs.astype(int)

    def predict(self, docs):
        return self.predict_proba(docs)[:, 1]


class DTree:
    def __init__(self, vectorizer, w_lists):
        # returns 1 if at least one list of w_lists is present in the example
        self.vectorizer = vectorizer
        self.w_lists = w_lists
        self.ids = {}
        for i in range(len(w_lists)):
            self.ids[i] = [vectorizer.vocabulary_[w] for w in w_lists[i]]

    # CLASSIFIER
    def predict_proba(self, docs):
        outs = np.zeros((len(docs), 2))
        vect = self.vectorizer.transform(docs)
        for i, x in enumerate(docs):
            if any([all([vect[i, j] > 0 for j in self.ids[l]]) for l in range(len(self.ids))]):
                outs[i, 1] = 1
        outs[:, 0] = 1 - outs[:, 1]
        return outs.astype(int)

    def predict(self, docs):
        return self.predict_proba(docs)[:, 1]


def load_models(dataset, model_name, docs, vectorization='tf_idf'):
    PATH = os.path.join(os.getcwd(), 'models')

    if model_name == 'nn_3':
        model = nn.Sequential(nn.Linear(1000, 500),
                              nn.ReLU(),

                              nn.Linear(500, 100),
                              nn.ReLU(),

                              nn.Linear(100, 2),
                              nn.LogSoftmax(dim=1))
        model.load_state_dict(torch.load(os.path.join(PATH, f'nn_3_{dataset}.p')))
        model.eval()

    if model_name == 'nn_10':
        model = nn.Sequential(nn.Linear(1000, 900),
                              nn.ReLU(),

                              nn.Linear(900, 800),
                              nn.ReLU(),

                              nn.Linear(800, 700),
                              nn.ReLU(),

                              nn.Linear(700, 600),
                              nn.ReLU(),

                              nn.Linear(600, 500),
                              nn.ReLU(),

                              nn.Linear(500, 400),
                              nn.ReLU(),

                              nn.Linear(400, 300),
                              nn.ReLU(),

                              nn.Linear(300, 200),
                              nn.ReLU(),

                              nn.Linear(200, 100),
                              nn.ReLU(),

                              nn.Linear(100, 2),
                              nn.LogSoftmax(dim=1))
        model.load_state_dict(torch.load(os.path.join(PATH, f'nn_10_{dataset}.p')))
        model.eval()

    if model_name == 'nn_20':
        model = nn.Sequential(nn.Linear(1000, 1000),
                              nn.ReLU(),

                              nn.Linear(1000, 950),
                              nn.ReLU(),

                              nn.Linear(950, 900),
                              nn.ReLU(),

                              nn.Linear(900, 850),
                              nn.ReLU(),

                              nn.Linear(850, 800),
                              nn.ReLU(),

                              nn.Linear(800, 750),
                              nn.ReLU(),

                              nn.Linear(750, 700),
                              nn.ReLU(),

                              nn.Linear(700, 650),
                              nn.ReLU(),

                              nn.Linear(650, 600),
                              nn.ReLU(),

                              nn.Linear(600, 550),
                              nn.ReLU(),

                              nn.Linear(550, 500),
                              nn.ReLU(),

                              nn.Linear(500, 450),
                              nn.ReLU(),

                              nn.Linear(450, 400),
                              nn.ReLU(),

                              nn.Linear(400, 350),
                              nn.ReLU(),

                              nn.Linear(350, 300),
                              nn.ReLU(),

                              nn.Linear(300, 250),
                              nn.ReLU(),

                              nn.Linear(250, 200),
                              nn.ReLU(),

                              nn.Linear(200, 150),
                              nn.ReLU(),

                              nn.Linear(150, 100),
                              nn.ReLU(),

                              nn.Linear(100, 50),
                              nn.ReLU(),

                              nn.Linear(50, 2),
                              nn.LogSoftmax(dim=1))
        model.load_state_dict(torch.load(os.path.join(PATH, f'nn_20_{dataset}.p')))
        model.eval()

    # redefine the classifiers, text as input
    def clf_proba(xi):
        z = torch.tensor(scipy.sparse.csr_matrix.todense(docs.vectorizer.transform(xi))).float()
        return torch.exp(model(torch.atleast_2d(z)))

    def clf(xi):
        return clf_proba(xi).argmax(1).numpy()

    return clf_proba, clf, model
