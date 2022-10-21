# dataset manager
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import scipy

import re

import os

datasets = ['restaurants', 'yelp', 'imdb']


class Dataset:
    def __init__(self, name, path):
        assert name in datasets, 'DATASET ' + str(name) + ' not available! Chose among ' + str(datasets)
        # light preprocessing: remove symbols, keep spaces, lowercase
        if name == 'restaurants':
            dataset = 'restaurants.tsv'
            self.df = pd.read_csv(os.path.join(path, dataset), sep='\t')
            self.X = [self.preprocess(x) for x in list(self.df["Review"])]
            self.y = list(self.df["Liked"])
        elif name == 'yelp':
            dataset = 'positive_negative_reviews_yelp.csv'
            self.df = pd.read_csv(os.path.join(path, dataset), sep='|')
            self.X = [self.preprocess(x) for x in list(self.df["text"])]
            self.y = list(self.df["stars"])

        elif name == 'imdb':
            dataset = 'imdb.csv'
            self.df = pd.read_csv(os.path.join(path, dataset), sep=',', nrows=10000)
            self.X = [self.preprocess(x) for x in list(self.df["review"])]
            self.y = self.df.sentiment.copy()
            self.y = self.y.replace('positive', 1)
            self.y = self.y.replace('negative', 0)
            self.y = list(self.y)

    def preprocess(self, x):
        return re.sub('[^a-zA-Z\d\s]', '', x).lower()


class Documents:
    def __init__(self, dataset, datapath, vectorization):
        data = Dataset(dataset, datapath)
        X, y = data.X, data.y

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        self.docs = X_test

        # Vectorization
        if vectorization == 'tf_idf':
            self.vectorizer = TfidfVectorizer(norm=None, max_features=1000)
        elif vectorization == 'norm_tf_idf':
            self.vectorizer = TfidfVectorizer(max_features=1000)
        self.train_vectors = self.vectorizer.fit_transform(X_train)
        self.test_vectors = self.vectorizer.transform(X_test)

        # convert to pytorch tensors
        self.train_tensors = torch.tensor(scipy.sparse.csr_matrix.todense(self.train_vectors)).float()
        self.test_tensors = torch.tensor(scipy.sparse.csr_matrix.todense(self.test_vectors)).float()

    def get_docs(self):
        docs_vectors = self.vectorizer.transform(self.docs)
        docs_x = torch.tensor(scipy.sparse.csr_matrix.todense(docs_vectors)).float()
        docs_x.requires_grad = True
        return self.docs, docs_x
