"""

Checking a conjecture about the distribution of linear combination of binomials with offset.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from numpy.matlib import repmat
from scipy.stats import norm

np.random.seed(0)

# specific parameters
sns.set_theme(style='darkgrid')
sns.set(font_scale=2)

# parameter of the Binomial
p = 0.5

# dimension of the model
dim = 10

# multiplicities of the words
multiplicities = np.array([10, 10, 8, 7, 6, 2, 2, 2, 2, 1])
# anchors
anchors = np.array([8, 5, 6, 5, 5, 2, 1, 1, 1, 1])

# \lambda_jv_j
coefs = 5 * np.random.normal(0, 1, (dim,))

# for the clt approximation
mean = 0.5 * np.dot(coefs, multiplicities + anchors)
std = np.sqrt(0.25 * np.dot(np.square(coefs), multiplicities - anchors))

# simulating \proba(X_1+\cdots+X_d \geq t)

#
n_simus = 100000
n_grid = 300
t_start = mean - 3 * std
t_end = mean + 3 * std
t_grid = np.linspace(t_start, t_end, n_grid)

est_proba = np.zeros((n_grid,))
for i_grid in range(n_grid):
    print("step {} / {}".format(i_grid, n_grid))
    t = t_grid[i_grid]
    m_array = repmat(anchors, n_simus, 1) + np.random.binomial(multiplicities - anchors, p, (n_simus, dim))
    # value_array = np.sum(m_array,1)
    value_array = np.dot(m_array, coefs)
    est_proba[i_grid] = np.mean(value_array <= t)

############################################################################

sns.lineplot(t_grid, est_proba, color='b', label='Estimated')
sns.lineplot(t_grid, norm.cdf((t_grid - mean) / std), color='r', label='Standard Normal')
plt.legend(loc='lower right')
plt.title(r'$d={}$, $b={}$, $|A|={}$'.format(dim, np.sum(multiplicities), np.sum(anchors)))

fig_name = "berry-esseen_{}_{}_{}.pdf".format(dim, np.sum(multiplicities), np.sum(anchors))
plt.savefig(fig_name, format='pdf', bbox_inches='tight', pad_inches=0)
