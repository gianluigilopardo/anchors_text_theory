"""

Checking the Berry-Esseen statement for normalized TF-IDF.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from numpy.matlib import repmat
from scipy.stats import norm


# specific parameters
sns.set_theme(style='darkgrid')
sns.set(font_scale=2)


SEED = 42
np.random.seed(SEED)

# specific parameters
sns.set_theme(style='darkgrid')
sns.set(font_scale=2)

# parameter of the Binomial
p = 0.5

# just different combinations of d, b, |A|
ds = [5, 20]
rs = [5, 10]
ps = [0.3, 0.7]

for d in ds:
    for r in rs:
        for ro in ps:
            multiplicities = np.round(r*(np.random.rand(d))).astype(int)
            anchors = [np.random.binomial(n, ro) for n in multiplicities]
            
            b = np.sum(multiplicities)
            a = np.sum(anchors)
            
            # dimension of the model
            dim = d
            
            
            # \lambda_j
            lambdas = np.random.normal(0, 1, (dim,))
            # lambdas = lambdas / np.linalg.norm(lambdas, 2)  # norm = 1
            # \v_j
            idfs = 7 - np.abs(np.random.normal(0, 1, (dim,)))
            # \lambda_j v_j
            coefs = lambdas*idfs
            
            # for the clt approximation
            mean_D = 1/4 * np.dot(np.square(multiplicities + anchors) + (multiplicities - anchors), np.square(idfs))
            mean_N = 1/2 * np.dot(coefs, multiplicities + anchors) 
            std_N = np.sqrt(1/4 * np.dot(np.square(coefs), multiplicities - anchors))
            
            # 
            n_simus = 100000
            n_grid = 300
            t_start = -3  
            t_end = +3  
            t_grid = np.linspace(t_start, t_end, n_grid)
            
            est_proba = np.zeros((n_grid,))
            for i_grid in range(n_grid):
                print("step {} / {}".format(i_grid, n_grid))
                t = t_grid[i_grid]
                m_array = repmat(anchors, n_simus, 1) + np.random.binomial(multiplicities - anchors, p, (n_simus, dim))
                value_array = np.dot(m_array, coefs) / np.linalg.norm(m_array*idfs, 2, 1)
                est_proba[i_grid] = np.mean(value_array <= t)
            
            ############################################################################
            
            sns.lineplot(t_grid, est_proba, color='b', label='Estimated')
            sns.lineplot(t_grid, norm.cdf((t_grid*np.sqrt(mean_D) - mean_N) / std_N), color='r', label='Standard Normal')
            plt.legend(loc='lower right')
            plt.title(r'$d={}$, $b={}$, $|A|={}$'.format(dim, np.sum(multiplicities), np.sum(anchors)))
            
            fig_name = f"norm_berry-esseen_{d}_{b}_{a}.pdf"
            plt.savefig(fig_name, format='pdf', bbox_inches='tight', pad_inches=0)
            plt.clf()
