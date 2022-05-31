"""
Computing Jaccard similarity between anchor A and first |A| words ranked by lambda_j*v_j
Logistic model
"""

import numpy as np
import pandas as pd

import spacy  # nlp for Anchors

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from anchor import anchor_text  # official Anchors

import os

from utils.general import rank_by_coefs
from dataset.dataset import Dataset
from utils.general import jaccard_similarity

import pickle
import time

# DATA
path = os.getcwd().replace('linear_models', '').replace('analysis', 'dataset')
DATASET = 'restaurants'
data = Dataset(DATASET, path)
df, X, y = data.df, data.X, data.y

X_train, X_test, y_train, y_test = train_test_split(X, y)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(norm=None)
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)

# CLASSIFIER
# any linear model
model = LogisticRegression()
model.fit(train_vectors, y_train)

# pipeline: Vectorizer + Model
c = make_pipeline(vectorizer, model)

y_pred = c.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# initialize the explainer
class_names = ["Dislike", "Like"]
nlp = spacy.load('en_core_web_sm')  # anchors need it
anchor_explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=True)


N_runs = 10

# We explain positive predictions
corpus = np.asarray(X_test)[c.predict(X_test) == 1]

# results
anchors_results = pd.DataFrame(columns=['Example', 'Anchor', 'Expected', 'Run', 'Similarity', 'Time'])

xi, anchors, expected, runs, similarity, times = [], [], [], [], [], []
for i, example in np.ndenumerate(corpus):
    for j in range(N_runs):
        t0 = time.time()
        print('\nRun: {} / {} - Example {} / {}: {}'.format(str(j+1), str(N_runs), str(i[0]+1), str(len(corpus)), example))
        print(c.predict_proba([example]))
        anchors_exp = anchor_explainer.explain_instance(str(example), c.predict).names()
        print(anchors_exp)
        anchors_exp.sort()
        word_coefs = rank_by_coefs(model, example, vectorizer)[:len(anchors_exp)]
        print(word_coefs)
        current_similarity = jaccard_similarity(anchors_exp, word_coefs)
        print(current_similarity)
        tf = time.time() - t0
        xi.append(example)
        anchors.append(anchors_exp)
        expected.append(word_coefs)
        runs.append(j)
        similarity.append(current_similarity)
        times.append(tf)


anchors_results.Example = xi
anchors_results.Anchor = anchors
anchors_results.Expected = expected
anchors_results.Run = runs
anchors_results.Similarity = similarity
anchors_results.Time = times
print(anchors_results)

filename = str('linear_' + str(DATASET) + '.p')
pickle.dump(anchors_results, open(filename, 'wb'))
