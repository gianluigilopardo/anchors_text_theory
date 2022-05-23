"""
Jaccard similarity between exhaustive-empirical Anchors and official implementation.
We test four models on two datasets.
"""

import empirical_anchor_text  # exhaustive-empirical Anchors
from anchor import anchor_text  # Official Anchors

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
import spacy
import pickle
import os
from utils import models
from utils.general import jaccard_similarity

from dataset.dataset import Dataset

nlp = spacy.load('en_core_web_sm')  # anchors need it

# DATA
path = os.getcwd().replace('exhaustive_anchors', 'dataset')
datasets = ['restaurants', 'yelp']
similarity = {}

for dataset in datasets:

    data = Dataset(dataset, path)
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y)

    # classes of the data
    class_names = ["Dislike", "Like"]

    # (non-normalized) TF-IDF
    vectorizer = TfidfVectorizer(norm=None)
    vectors = vectorizer.fit_transform(X_train)

    # MODELS
    # model returning 1 if 'good' is present
    indicator = models.Product(vectorizer, ['good'])
    # model returning 1 if 'not' and 'bad are present or 'good' is present
    dtree = models.DTree(vectorizer, [['not', 'bad'], ['good']])
    # logistic model
    logistic = make_pipeline(vectorizer, LogisticRegression().fit(vectors, y_train))
    # perceptron
    perceptron = make_pipeline(vectorizer, Perceptron().fit(vectors, y_train))
    # random forest classifier
    rf = make_pipeline(vectorizer, RandomForestClassifier().fit(vectors, y_train))
    models_data = {'Indicator': indicator, 'DTree': dtree, 'LogisticRegression': logistic, 'Perceptron': perceptron, 'RandomForestClassifier': rf}

    # Jaccard similarity
    data_similarity = {'Indicator': 0, 'DTree': 0, 'LogisticRegression': 0, 'Perceptron': 0, 'RandomForestClassifier': 0}
    for k, model in models_data.items():
        # initialize the explainers
        exhaustive_explainer = empirical_anchor_text.AnchorText(model.predict)
        official_explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=True)
        # We explain posite predictions
        X = np.asarray(X_test)[model.predict(X_test) == 1]
        for i, doc in enumerate(X):
            print('\n {} / {} of {}'.format((i+1), len(X), dataset))
            print(doc)
            prediction = model.predict([doc])
            print('{} predicts {}'.format(k, str(prediction)))
            exhaustive_anchor = exhaustive_explainer.explain_instance(doc)
            official_anchor = official_explainer.explain_instance(str(doc), model.predict).names()
            current_similarity = jaccard_similarity(official_anchor, exhaustive_anchor)
            data_similarity[k] = data_similarity[k] + current_similarity
            print('Official anchor: {} \nExhaustive anchor: {} \nSimilarity: {}'.format(
                official_anchor, exhaustive_anchor, current_similarity))
        data_similarity[k] = data_similarity[k] / len(X)
    similarity[dataset] = data_similarity

results = pd.DataFrame(columns=list(models_data.keys()), index=datasets)

for dataset in datasets:
    for k, model in models_data.items():
        results.loc[dataset][k] = similarity[dataset][k]

print(results)
pickle.dump(results, open('similarity.p', 'wb'))
