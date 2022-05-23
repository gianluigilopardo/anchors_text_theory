"""
Simple classifier relying on the presence of one given words
Product of indicator functions, returning 1 if "very" and "good" are present

"""


from anchor import anchor_text  # Official Anchors

from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pickle
import os
from utils import models
from utils.general import count_multiplicities

from dataset.dataset import Dataset


# DATA
path = os.getcwd().replace('simple_rules', '').replace('analysis', 'dataset')
data = Dataset('yelp', path)
df, X, y = data.df, data.X, data.y

N_runs = 100

# TF-IDF transformation
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(X)

# MODELS
# model returning 1 if 'good' and 'very' are present
model = models.Product(vectorizer, ['very', 'good'])

# classes of the model
class_names = ["Dislike", "Like"]

# initialize the explainers
nlp = spacy.load('en_core_web_sm')  # anchors need it
anchor_explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=True)

# Example to explain
example = "Food is very very very very good!"
print(example)
anchors_res = []
for i in range(N_runs):
    print('\n {} / {}'.format((i + 1), N_runs))
    anchors_exp = anchor_explainer.explain_instance(example, model.predict).names()
    print(anchors_exp)
    anchors_exp.sort()
    anchor_as_string = ', '.join(w for w in anchors_exp)
    anchors_res.append(anchor_as_string)

print(anchors_res)

multiplicities = count_multiplicities(example)
info = {'Description': 'Classifier returning 1 if \'good\' and \'very\' are present\n'
                       'Limit case: the multiplicity of \'very\' in the example is equal to the breakpoint value B=4',
        'Example': example, 'N_runs': N_runs,
        'Multiplicities': multiplicities,
        'Anchors': anchors_res}
path = os.getcwd().replace('analysis', 'results')
pickle.dump(info, open(os.path.join(path, str('presence_words_B.p')), 'wb'))


