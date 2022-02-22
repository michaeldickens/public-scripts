"""

mission_hedging.py
------------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2022-02-21

"""

import numpy as np
import pymc3 as pm
import pymc3.distributions.transforms as tr
import theano.tensor as tt


class Exp(tr.ElemwiseTransform):
    name = "exp"

    def backward(self, x):
        return tt.log(x)

    def forward(self, x):
        return tt.exp(x)

    def jacobian_det(self, x):
        return -tt.log(x)


with pm.Model():
    means = np.array([
        0.08,  # index return
        {      # beta-neutral hedge return
            'NoDur':  0.07,
            'Hlth':   0.04,
            'HiTec': -0.02,
        }['Hlth'],
        0.1,  # co2 increase
    ])

    stdevs = [0.18, 0.18, 0.05]
    correlations = [
        [1, 0  , 0  ],
        [0, 1  , 0.5],
        [0, 0.5, 1  ],
    ]
    covariances = np.array([
        [correl * stdev1 * stdev2 for correl, stdev2 in zip(row, stdevs)]
        for row, stdev1 in zip(correlations, stdevs)
    ])

    vals = pm.MvNormal('vals', mu=means, cov=covariances)

    money = (1 - prop_hedge) * tt.exp(vals[0]) + prop_hedge * tt.exp(vals[1])
    total_co2 = tt.exp(vals[2])

    utility = tt.log(money) * total_co2
