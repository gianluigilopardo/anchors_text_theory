"""
Small neural network
"""

# import
import numpy as np
import pandas as pd

import spacy  # nlp for Anchors

import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import scipy
from torch import optim
from torch.autograd import Variable
import logging

import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
# torch.__version__
from anchor import anchor_text  # official Anchors

import time
import os

from utils.general import get_coefficients_nn
from utils.general import count_multiplicities
from utils.general import get_idf
from dataset.dataset import Dataset
from utils.general import jaccard_similarity
from utils.general import rank_by_coefs_nn

# output
import pickle

np.random.seed(10)

# DATA
path = os.getcwd().replace('nn_gradient', '').replace('analysis', 'dataset')
DATASET = 'imdb'
data = Dataset(DATASET, path)
df, X, y = data.df, data.X, data.y

X_train, X_test, y_train, y_test = train_test_split(X, y)

# norm TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vect = 'norm_tf_idf'

train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)

# convert to pytorch tensor
x_train = torch.tensor(scipy.sparse.csr_matrix.todense(train_vectors)).float()
x_test = torch.tensor(scipy.sparse.csr_matrix.todense(test_vectors)).float()
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

TRAIN = True
PATH = os.path.join(os.getcwd(), vect, f'net_{DATASET}')

if TRAIN:
    # CLASSIFIER
    model = nn.Sequential(nn.Linear(x_train.shape[1], 128),
                          nn.ReLU(),

                          nn.Linear(128, 64),
                          nn.ReLU(),

                          nn.Linear(64, 2),
                          nn.Softmax(dim=1))

    # Define the loss
    criterion = nn.NLLLoss()

    # Forward pass, get our logits
    logps = model(x_train)
    # Calculate the loss with the logits and the labels
    loss = criterion(logps, y_train)

    loss.backward()

    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    # train
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    epochs = 1000
    for e in range(epochs):
        optimizer.zero_grad()
        output = model.forward(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        train_loss = loss.item()
        train_losses.append(train_loss)
        optimizer.step()

        ps = torch.exp(model(x_train))
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == y_train.view(*top_class.shape)
        train_accuracy = torch.mean(equals.float())
        train_accuracies.append(train_accuracy)

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            log_ps = model(x_test)
            test_loss = criterion(log_ps, y_test)
            test_losses.append(test_loss)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == y_test.view(*top_class.shape)
            test_accuracy = torch.mean(equals.float())
            test_accuracies.append(test_accuracy)

        model.train()

        print(f"Epoch: {e + 1}/{epochs}.. ",
              f"Training Loss: {train_loss:.3f}.. ",
              f"Training Accuracy: {train_accuracy:.3f}.. ",
              f"Test Loss: {test_loss:.3f}.. ",
              f"Test Accuracy: {test_accuracy:.3f}")

    torch.save(model.state_dict(), PATH)

else:
    model = nn.Sequential(nn.Linear(x_train.shape[1], 128),
                          nn.ReLU(),

                          nn.Linear(128, 64),
                          nn.ReLU(),

                          nn.Linear(64, 2),
                          nn.Softmax(dim=1))
    criterion = nn.NLLLoss()
    model.load_state_dict(torch.load(PATH))
    model.eval()


# redefine the classifiers, text as input
def clf_proba(xi):
    z = torch.tensor(scipy.sparse.csr_matrix.todense(vectorizer.transform(xi))).float()
    return model(torch.atleast_2d(z))


def clf(xi):
    return clf_proba(xi).argmax(1).numpy()


y_pred = clf(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

info = {'network': '128-64-2',
        'loss': 'nn.NLLLoss()',
        'test_accuracy': accuracy}

# initialize the explainer
class_names = ['negative', 'positive']
nlp = spacy.load('en_core_web_sm')  # anchors need it
anchor_explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=True)

N_runs = 10

# We explain positive predictions
corpus = np.asarray(X_test)[clf(X_test) == 1][:100]
corpus_vectors = vectorizer.transform(corpus)
corpus_x = torch.tensor(scipy.sparse.csr_matrix.todense(corpus_vectors)).float()

corpus_x.requires_grad = True
y = model(corpus_x)
pred = y.argmax(dim=1)
loss = criterion(y, pred)
loss.backward()

# results
anchors_results = pd.DataFrame(columns=['Example', 'Proba', 'Anchor', 'Expected', 'Run', 'Similarity', 'Time'])

xi, probas, anchors, expected, runs, similarity, times = [], [], [], [], [], [], []
for i, example in np.ndenumerate(corpus):
    idx = i[0]
    for j in range(N_runs):
        t0 = time.time()
        print('\nRun: {} / {} - Example {} / {}: {}'.format(str(j + 1), str(N_runs), str(idx + 1), str(len(corpus)),
                                                            example))
        proba = clf_proba([example])[:, 1].detach().numpy()[0]
        print(proba)
        probas.append(proba)
        anchors_exp = anchor_explainer.explain_instance(str(example), clf).names()
        print(anchors_exp)
        anchors_exp.sort()
        gradient = corpus_x.grad[idx]
        word_coefs = rank_by_coefs_nn(gradient, example, vectorizer)[:len(anchors_exp)]
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
anchors_results.Proba = probas
anchors_results.Anchor = anchors
anchors_results.Expected = expected
anchors_results.Run = runs
anchors_results.Similarity = similarity
anchors_results.Time = times
print(anchors_results)

res = os.path.join(os.getcwd(), vect)
pickle.dump(anchors_results, open(os.path.join(res, f'{DATASET}_similarity.p'), 'wb'))
pickle.dump(info, open(os.path.join(res, f'{DATASET}_info.p'), 'wb'))

