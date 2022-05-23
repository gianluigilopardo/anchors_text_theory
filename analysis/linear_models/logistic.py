"""
Logistic model
"""

# import
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

from utils.general import get_coefficients
from utils.general import count_multiplicities
from utils.general import get_idf
from dataset.dataset import Dataset

# output
import pickle

np.random.seed(0)

# DATA
path = os.getcwd().replace('linear_models', '').replace('analysis', 'dataset')
DATASET = 'yelp'
data = Dataset(DATASET, path)
df, X, y = data.df, data.X, data.y

X_train, X_test, y_train, y_test = train_test_split(X, y)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(norm=None)
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)

# CLASSIFIER
model = LogisticRegression()
model.fit(train_vectors, y_train)

# pipeline: Vectorizer + Model
c = make_pipeline(vectorizer, model)

y_pred = c.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

example = X[13]
print(example)
print(c.predict_proba([example]))

N_runs = 10

anchors_data = pd.DataFrame(columns=['Anchor', 'Run', 'Shift'])
runs, shifts_, anchors = [], [], []

# classes of the model
class_names = ["Dislike", "Like"]

# initialize the explainers
nlp = spacy.load('en_core_web_sm')  # anchors need it
anchor_explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=True)

shifts = np.linspace(0.0, 5.4, 55)
intercept = model.intercept_[0]

p = vectorizer.build_preprocessor()
p_doc = p(example)
t = vectorizer.build_tokenizer()
words = t(p_doc)

multiplicities = count_multiplicities(example, vectorizer)
coefficients = get_coefficients(model, example, vectorizer)
idf = get_idf(example, vectorizer)
mv = {w: coefficients[w]*idf[w] for w in words}
mv = dict(sorted(mv.items(), key=lambda item: item[1], reverse=True))

print(mv)
for shift in shifts:
    model.intercept_ = [intercept - shift]
    print(model.intercept_)
    anchors_res = []
    for i in range(N_runs):
        print('\nRun: {} / {} - Shift: {}'.format(str(i+1), str(N_runs), shift))
        print(c.predict_proba([example]))
        anchors_exp = anchor_explainer.explain_instance(example, c.predict).names()
        print(anchors_exp)
        anchors_exp.sort()
        anchor_as_string = ', '.join(w for w in anchors_exp)
        anchors_res.append(anchor_as_string)
        for ele in set(anchors_res):
            anchors.append(ele)
            shifts_.append(shift)
            runs.append(i)

anchors_data.Anchor = anchors
anchors_data.Run = runs
anchors_data.Shift = shifts_


info = {'Description': 'Anchors explanation for a logistic model',
        'Dataset': DATASET, 'Accuracy': accuracy, 'Confusion Matrix': cm,
        'Example': example, 'N_runs': N_runs,
        'Coefficients': coefficients, 'Multiplicities': multiplicities, 'IDF': idf, 'mv': mv,
        'Intercept': intercept, 'Shifts': shifts,
        'Anchors': anchors_data,
        }

print()
print(anchors_data)
path = os.getcwd().replace('analysis', 'results')
pickle.dump(info, open(os.path.join(path, str('logistic.p')), 'wb'))

