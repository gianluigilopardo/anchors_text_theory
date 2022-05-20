"""
This is the exhaustive-empirical implementation of Anchors for text data
"""

import numpy as np
from itertools import chain, combinations


class AnchorText:

    def __init__(self, classifier_fn, num_samples=1000, threshold=0.95, mask='UNK'):
        """
        :param classifier_fn: classifier prediction function, taking a list of documents as input
        :param num_samples: size of the neighborhood to learn the anchor
        :param threshold: minimal precision to select an anchor
        :param mask: token to replace words while sampling
        """
        self.classifier_fn = classifier_fn
        self.threshold = threshold
        self.mask = mask
        self.num_samples = num_samples

    def compute_precision(self, sample, prediction):
        """
        :param sample: perturbed neighborhood of the example to be explained
        :param prediction: output of the classifier of the example
        :return: proportion of the sample with same prediction as the example
        """
        preds = self.classifier_fn(sample)
        precision = 1 / len(sample) * np.sum(preds == prediction)
        return precision

    def sample_fn(self, words, present):
        """
        :param words: list of words in a document
        :param present: index of words in the anchor
        :return: num_samples perturbed version of the example
        """
        num_samples = self.num_samples
        data = np.ones((num_samples, len(words)))
        raw = np.zeros((num_samples, len(words)), '|U80')
        raw[:] = words
        for i, t in enumerate(words):
            if i in present:
                continue
            n_changed = np.random.binomial(num_samples, .5)
            changed = np.random.choice(num_samples, n_changed, replace=False)
            raw[changed, i] = self.mask
            data[changed, i] = 0
        raw_data = [' '.join(x) for x in raw]
        return raw_data

    def explain_instance(self, text):
        """
        :param text: example to be explained
        :return: shortest anchor with maximal precision
        """
        threshold = self.threshold
        words = text.split()
        prediction = self.classifier_fn([text])[0]
        candidates = chain.from_iterable(combinations(words, r) for r in range(len(words) + 1))
        precision_best = 0
        len_best = 0
        anchor = []  # best candidate
        for a in candidates:
            a = list(a)
            if not a:
                continue
            present = [words.index(a[i]) for i in range(len(a))]
            if precision_best >= threshold and len(a) > len_best:
                return anchor
            sample = self.sample_fn(words, present)
            precision_current = self.compute_precision(sample, prediction)
            if precision_current == precision_best and len(a) <= len(anchor):
                precision_best = precision_current
                len_best = len(a)
                anchor = a
            if precision_current >= precision_best:
                precision_best = precision_current
                len_best = len(a)
                anchor = a
        return anchor
