"""
Dataset manager
"""

import pandas as pd
import os

datasets = ['restaurants', 'yelp']


class Dataset:
    def __init__(self, name, path):
        assert name in datasets, 'DATASET {} not available! Choose among {}'.format(name, datasets)
        if name == 'restaurants':
            dataset = 'restaurants.tsv'
            self.df = pd.read_csv(os.path.join(path, dataset), sep='\t')
            self.X = list(self.df["Review"])
            self.y = list(self.df["Liked"])
        elif name == 'yelp':
            dataset = 'positive_negative_reviews_yelp.csv'
            self.df = pd.read_csv(os.path.join(path, dataset), sep='|')
            self.X = list(self.df["text"])
            self.y = list(self.df["stars"])

