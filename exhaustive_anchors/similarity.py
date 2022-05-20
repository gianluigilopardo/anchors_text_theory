"""
Jaccard similarity between exhaustive-empirical Anchors and official implementation.
We test four models on two datasets.
"""

import empirical_anchor_text  # exhaustive-empirical Anchors
from anchor import anchor_text  # Official Anchors

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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
data_yelp = Dataset('yelp', path)
X_train_yelp, X_test_yelp, y_train_yelp, y_test_yelp = train_test_split(data_yelp.X, data_yelp.y)
data_restaurants = Dataset('restaurants', path)
X_train_restaurants, X_test_restaurants, y_train_restaurants, y_test_restaurants = train_test_split(data_restaurants.X, data_restaurants.y)

# classes of the data
class_names = ["Dislike", "Like"]

# (non-normalized) TF-IDF
vectorizer_yelp = TfidfVectorizer(norm=None)
vectorizer_restaurants = TfidfVectorizer(norm=None)
vectors_yelp = vectorizer_yelp.fit_transform(X_train_yelp)
vectors_restaurants = vectorizer_restaurants.fit_transform(X_train_restaurants)

# MODELS
# model returning 1 if 'good' is present
indicator_yelp = models.Product(vectorizer_yelp, ['good'])
indicator_restaurants = models.Product(vectorizer_restaurants, ['good'])
# model returning 1 if 'not' and 'bad are present or 'good' is present
dtree_yelp = models.DTree(vectorizer_yelp, [['not', 'bad'], ['good']])
dtree_restaurants = models.DTree(vectorizer_restaurants, [['not', 'bad'], ['good']])
# logistic model
logistic_yelp = LogisticRegression().fit(vectors_yelp, y_train_yelp)
logistic_restaurants = LogisticRegression().fit(vectors_restaurants, y_train_restaurants)
# perceptron
perceptron_yelp = Perceptron().fit(vectors_yelp, y_train_yelp)
perceptron_restaurants = Perceptron().fit(vectors_restaurants, y_train_restaurants)
# random forest classifier
rf_yelp = RandomForestClassifier().fit(vectors_yelp, y_train_yelp)
rf_restaurants = RandomForestClassifier().fit(vectors_restaurants, y_train_restaurants)


# Jaccard similarity for Yelp
similarity_yelp = {'Indicator': 0, 'DTree': 0, 'LogisticRegression': 0, 'Perceptron': 0, 'RandomForestClassifier': 0}
models_yelp = {'Indicator': indicator_yelp, 'DTree': dtree_yelp, 'LogisticRegression': logistic_yelp, 'Perceptron': perceptron_yelp, 'RandomForestClassifier': rf_yelp}
for k, model in models_yelp.items():
    # initialize the explainers
    exhaustive_explainer = empirical_anchor_text.AnchorText(model.predict)
    official_explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=True)
    X = np.asarray(X_test_yelp)[model.predict(X_test_yelp) == 1]
    for i, doc in enumerate(X):
        print('\n {} / {}'.format((i+1), len(X)))
        print(doc)
        prediction = model.predict([doc])
        print('{} predicts {}'.format(k, str(prediction)))
        exhaustive_anchor = exhaustive_explainer.explain_instance(doc)
        official_anchor = official_explainer.explain_instance(str(doc), model.predict).names()
        current_similarity = jaccard_similarity(official_anchor, exhaustive_anchor)
        similarity_yelp[k] = similarity_yelp[k] + current_similarity
        print('Official anchor: {} \nExhaustive anchor: {} \nSimilarity: {} \nOverall similarity: {}'.format(
            official_anchor, exhaustive_anchor, current_similarity, similarity_yelp))
    similarity_yelp[k] = similarity_yelp[k] / len(X)

# Jaccard similarity for Restaurants
similarity_restaurants = {'Indicator': 0, 'DTree': 0, 'LogisticRegression': 0, 'Perceptron': 0, 'RandomForestClassifier': 0}
models_restaurants = {'Indicator': indicator_restaurants, 'DTree': dtree_restaurants, 'LogisticRegression': logistic_restaurants,
               'Perceptron': perceptron_restaurants, 'RandomForestClassifier': rf_restaurants}
for k, model in models_restaurants.items():
    # initialize the explainers
    exhaustive_explainer = empirical_anchor_text.AnchorText(model)
    official_explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=True)
    X = np.asarray(X_test_restaurants)[model.predict(X_test_restaurants)]
    for i, doc in enumerate(X):
        print('\n {} / {}'.format((i + 1), len(X)))
        print(doc)
        prediction = model.predict([doc])
        print('{} predicts {}'.format(k, prediction))
        exhaustive_anchor = exhaustive_explainer.explain_instance(doc)
        official_anchor = official_explainer.explain_instance(str(doc), model.predict).names()
        current_similarity = jaccard_similarity(official_anchor, exhaustive_anchor)
        similarity_restaurants[k] = similarity_restaurants[k] + current_similarity
        print('Official anchor: {} \nExhaustive anchor: {} \nSimilarity: {} \nOverall similarity: {}'.format(
            official_anchor, exhaustive_anchor, current_similarity, similarity_restaurants))
    similarity_restaurants[k] = similarity_restaurants[k] / len(X)


results = pd.DataFrame(columns=list(models_yelp.keys()), index=['Yelp', 'Restaurants'])
for k, model in models_yelp.items():
    results.loc['Yelp'][k] = similarity_yelp[k]
    results.loc['Restaurants'][k] = similarity_restaurants[k]

print(results)
pickle.dump(results, open('similarity.p', 'wb'))
