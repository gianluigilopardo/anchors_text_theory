import os
import pickle
from matplotlib import pyplot as plt
import seaborn as sns

# specific parameters
sns.set_theme(style='darkgrid')
lw = 3  # linewidth
ds = 5  # dot size
sns.set(font_scale=2)

models = {'linear_models': ['logistic', 'perceptron'],
          'simple_rules': ['dtree', 'presence_words_B', 'presence_words_break']}

for model in models['simple_rules']:
    print('Saving {}...'.format(model))
    info = pickle.load(open(os.path.join('results', 'simple_rules', str(model) + '.p'), 'rb'))
    anchors_res = info['Anchors']
    # Figure
    plt.tight_layout()
    plt.subplots(figsize=(10, 2))
    plt.title(info['Example'])
    filename_anchors = os.path.join('results', 'simple_rules', str(model) + '.pdf')
    sns.histplot(y=anchors_res, hue=anchors_res, alpha=1)
    plt.legend().remove()
    plt.savefig(filename_anchors, bbox_inches='tight', pad_inches=0)

for model in models['linear_models']:
    print('Saving {}...'.format(model))
    info = pickle.load(open(os.path.join('results', 'linear_models', str(model) + '.p'), 'rb'))
    n_words = 3
    words = list(info['mv'].keys())[:n_words]
    # use the following instead if re-running the experiments
    # words = list(info['coefs'].keys())[:n_words]
    a = [[] for i in range(n_words)]
    anchors_res = info['Anchors']
    for i, row in anchors_res.iterrows():
        for j in range(n_words):
            a[j].append(row.Anchor.count(words[j]))
    # Figure
    plt.tight_layout()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for j in range(n_words):
        lbl = r'{}'.format(words[j])
        sns.lineplot(data=anchors_res, x='Shift', y=a[j], label=lbl, linewidth=lw, markersize=ds)
    x = anchors_res['Shift']
    plt.xticks(rotation=45)  # Rotates X-Axis Ticks by 45-degrees
    plt.xlabel('Shift')
    plt.ylabel('Frequency')
    filename_anchors = os.path.join('results', 'linear_models', str(model) + '.pdf')
    plt.savefig(fname=filename_anchors, bbox_inches='tight', pad_inches=0)