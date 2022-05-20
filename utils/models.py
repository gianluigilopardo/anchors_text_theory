"""
Implementation of simple, explainable models
"""

import numpy as np


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

