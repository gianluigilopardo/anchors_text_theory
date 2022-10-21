import numpy as np
import torch
from torch import nn
import pandas as pd
import scipy

import spacy  # nlp for Anchors
from anchor import anchor_text  # official Anchors

from utils.models import load_models
from utils.general import jaccard_similarity
from utils.general import rank_by_coefs_nn
from dataset.dataset import Documents

import pickle
import time
import os

np.random.seed(42)
torch.manual_seed(42)

vectorization = 'tf_idf'
dir = os.path.join('analysis', 'nn_gradient')
datapath = os.getcwd().replace(dir, 'dataset')

#########################################################################################################

datasets = ['restaurants', 'yelp', 'imdb']
models = ['nn_3', 'nn_10', 'nn_20']

# initialize the explainer
class_names = ['negative', 'positive']
nlp = spacy.load('en_core_web_sm')  # anchors need it
anchor_explainer = anchor_text.AnchorText(nlp, class_names)

for dataset in datasets:
    for model_name in models:
        docs_obj = Documents(dataset, datapath, vectorization=vectorization)
        clf_proba, clf, model = load_models(dataset, model_name, docs_obj, vectorization=vectorization)
        docs, docs_torch = docs_obj.get_docs()

        # We explain positive predictions
        positive_docs = np.asarray(docs)[clf(docs) == 1]
        positive_vectors = docs_obj.vectorizer.transform(positive_docs)
        positive_tensors = torch.tensor(scipy.sparse.csr_matrix.todense(positive_vectors)).float()

        positive_tensors.requires_grad = True
        y = torch.exp(model(positive_tensors))
        pred = y.argmax(dim=1)
        criterion = nn.NLLLoss()
        loss = criterion(y, pred)
        loss.backward()

        # results
        anchors_results = pd.DataFrame(columns=['Example', 'Proba', 'Anchor', 'Expected', 'Run', 'Similarity', 'Time'])

        N_runs = 10
        xi, probas, anchors, expected, runs, similarity, times = [], [], [], [], [], [], []
        for i, example in np.ndenumerate(positive_docs):
            idx = i[0]
            for j in range(N_runs):
                t0 = time.time()
                print(f"\nModel: {model_name} - dataset: {dataset} - Run: {str(j + 1)} / {str(N_runs)} - "
                      f"Example {str(idx + 1)} / {str(len(positive_docs))}: {example}")
                proba = clf_proba([example])[:, 1].detach().numpy()[0]
                print(proba)
                probas.append(proba)
                anchors_exp = anchor_explainer.explain_instance(str(example), clf).names()
                print(anchors_exp)
                anchors_exp.sort()
                gradient = positive_tensors.grad[idx]
                word_coefs = rank_by_coefs_nn(gradient, example, docs_obj.vectorizer)[:len(anchors_exp)]
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

        res = os.path.join(datapath.replace('dataset', 'results'), 'nn_gradient')
        pickle.dump(anchors_results, open(os.path.join(res, f'{dataset}_{model_name}_similarity.p'), 'wb'))
