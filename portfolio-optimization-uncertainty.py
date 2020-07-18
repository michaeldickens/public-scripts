"""

portfolio-optimization-uncertainty.py
-------------------------------------

Author: Michael Dickens <mdickens93@gmail.com>
Created: 2020-07-09

Portfolio optimization when means and covariances are uncertain. Goal is to
maximize Sharpe ratio, not find the efficient frontier.

"""

import numpy as np
from scipy import optimize


def maximize_sharpe(means: np.array, cov_matrix: np.array) -> np.array:
    rf = means[0]  # by convention, first asset in the list should be risk free

    def negative_sharpe(weights: np.array):
        mean = np.dot(means, weights)
        stdev = np.sqrt(np.dot(np.dot(np.transpose(weights), cov_matrix), weights))
        return -(mean - rf) / stdev

    initial_guess = np.array([1 / len(means)] * len(means))
    sum_constraint = optimize.LinearConstraint([1] * len(means), lb=[1], ub=[1])
    opt = optimize.minimize(negative_sharpe, initial_guess, method='SLSQP', constraints=[sum_constraint])
    return opt.x


means = np.array([0.01, 0.02, 0.05])
cov_matrix = np.array([
    [0.03**2, 0, 0],
    [0, 0.05**2, 0],
    [0, 0, 0.15**2],
])

weights = maximize_sharpe(means, cov_matrix)
mean = np.dot(means, weights)
stdev = np.sqrt(np.dot(np.dot(np.transpose(weights), cov_matrix), weights))

print(weights)
print("mean {:.1f}%, stdev {:.1f}%, Sharpe {:.2f}".format(
    100 * mean,
    100 * stdev,
    (mean - means[0]) / stdev,
))
