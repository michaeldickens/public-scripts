"""

mvo.py
------

Author: Michael Dickens <mdickens93@gmail.com>
Created: 2020-10-05

"""

from pprint import pprint

import numpy as np
from scipy import optimize


covariances = None

# Returns and covariances downloaded from Research Affiliates' Asset Allocation
# Interactive on 2020-10-05.
#
# I originally tried to do this on the .xslx that I downloaded from RAFI, but
# LibreOffice's solver apparently sucks at solving optimization problems.
rafi_data = dict(
    asset_classes = [
        "US Large",
        "US Small",
        "All country",
        "EAFE",
        "Emerging Markets",
        "US Treasury Short",
        "US Treasury Long",
        "US Treasury Intermediate",
        "US Corporate Intermediate",
        "US High Yield",
        "US Aggregate",
        "US Tips",
        "Global Ex-US Treasury",
        "Global Aggregate",
        "Emerging Market (Non-Local)",
        "Emerging Market (Local)",
        "EM Cash",
        "Commodities",
        "Bank Loans",
        "Global Ex-US Corporates",
        "REITs",
        "United States Cash",
        "US Commercial Real Estate",
        "Global DM Ex-US Long/Short Equity",
        "US Long/Short Equity",
        "Europe LBO",
        "US LBO",
    ],

    means = np.array([
         0.2,  # US Large
         1.9,  # US Small
         2.7,  # All country
         5.0,  # EAFE
         6.9,  # Emerging Markets
        -0.8,  # US Treasury Short
        -3.8,  # US Treasury Long
        -0.7,  # US Treasury Intermediate
        -0.5,  # US Corporate Intermediate
         0.8,  # US High Yield
        -1.2,  # US Aggregate
        -0.4,  # US Tips
        -0.8,  # Global Ex-US Treasury
        -0.7,  # Global Aggregate
         0.8,  # Emerging Market (Non-Local)
         4.3,  # Emerging Market (Local)
         2.8,  # EM Cash
         1.0,  # Commodities
         0.7,  # Bank Loans
        -0.8,  # Global Ex-US Corporates
         1.5,  # REITs
        -0.6,  # United States Cash
         2.3,  # US Commercial Real Estate
         3.2,  # Global DM Ex-US Long/Short Equity
         1.0,  # US Long/Short Equity
         6.8,  # Europe LBO
         0.0,  # US LBO
    ]),

    covariances = np.array([
        [2.075, 2.484, 2.143, 2.136, 2.318, -0.060, -0.520, -0.144, 0.178, 0.905, -0.043, 0.092, 0.178, 0.194, 0.582, 0.997, 0.658, 0.856, 0.567, 0.699, 2.277, -0.011, 0.371, 1.054, 0.779, 2.557, 2.660],
        [2.484, 3.592, 2.523, 2.467, 2.718, -0.098, -0.842, -0.228, 0.150, 1.130, -0.121, 0.028, 0.087, 0.129, 0.615, 1.114, 0.763, 0.998, 0.712, 0.724, 2.930, -0.021, 0.481, 1.314, 1.050, 2.927, 3.161],
        [2.143, 2.523, 2.409, 2.537, 2.902, -0.059, -0.526, -0.141, 0.259, 1.064, -0.013, 0.167, 0.330, 0.326, 0.769, 1.344, 0.865, 1.098, 0.655, 0.943, 2.394, -0.011, 0.361, 1.298, 0.786, 2.986, 2.682],
        [2.136, 2.467, 2.537, 2.827, 3.088, -0.057, -0.528, -0.136, 0.307, 1.132, 0.005, 0.194, 0.426, 0.411, 0.864, 1.526, 0.970, 1.187, 0.684, 1.108, 2.416, -0.010, 0.319, 1.465, 0.773, 3.328, 2.638],
        [2.318, 2.718, 2.902, 3.088, 4.389, -0.062, -0.539, -0.140, 0.406, 1.401, 0.039, 0.350, 0.594, 0.552, 1.167, 2.093, 1.319, 1.692, 0.860, 1.334, 2.666, -0.016, 0.424, 1.664, 0.816, 3.485, 2.776],
        [-0.060, -0.098, -0.059, -0.057, -0.062, 0.035, 0.153, 0.056, 0.037, -0.036, 0.054, 0.047, 0.062, 0.051, 0.029, 0.007, -0.005, -0.051, -0.052, 0.019, -0.030, 0.018, -0.050, -0.018, -0.019, -0.067, -0.072],
        [-0.520, -0.842, -0.526, -0.528, -0.539, 0.153, 1.430, 0.328, 0.231, -0.194, 0.371, 0.397, 0.400, 0.310, 0.228, -0.052, -0.128, -0.286, -0.266, 0.052, 0.093, 0.049, -0.150, -0.228, -0.211, -0.683, -0.683],
        [-0.144, -0.228, -0.141, -0.136, -0.140, 0.056, 0.328, 0.110, 0.071, -0.071, 0.106, 0.109, 0.137, 0.109, 0.063, 0.024, -0.007, -0.070, -0.102, 0.042, -0.047, 0.024, -0.081, -0.041, -0.055, -0.176, -0.181],
        [0.178, 0.150, 0.259, 0.307, 0.406, 0.037, 0.231, 0.071, 0.186, 0.221, 0.122, 0.165, 0.177, 0.159, 0.253, 0.245, 0.127, 0.115, 0.105, 0.225, 0.370, 0.017, -0.039, 0.165, 0.053, 0.298, 0.157],
        [0.905, 1.130, 1.064, 1.132, 1.401, -0.036, -0.194, -0.071, 0.221, 0.782, 0.034, 0.181, 0.155, 0.161, 0.505, 0.636, 0.372, 0.554, 0.506, 0.455, 1.390, -0.016, 0.177, 0.565, 0.317, 1.178, 0.966],
        [-0.043, -0.121, -0.013, 0.005, 0.039, 0.054, 0.371, 0.106, 0.122, 0.034, 0.131, 0.146, 0.160, 0.134, 0.148, 0.093, 0.030, -0.023, -0.028, 0.103, 0.132, 0.023, -0.068, 0.019, -0.023, -0.028, -0.082],
        [0.092, 0.028, 0.167, 0.194, 0.350, 0.047, 0.397, 0.109, 0.165, 0.181, 0.146, 0.299, 0.245, 0.206, 0.275, 0.272, 0.132, 0.215, 0.062, 0.229, 0.374, 0.010, -0.002, 0.149, 0.018, 0.128, 0.048],
        [0.178, 0.087, 0.330, 0.426, 0.594, 0.062, 0.400, 0.137, 0.177, 0.155, 0.160, 0.245, 0.580, 0.454, 0.348, 0.597, 0.350, 0.340, -0.026, 0.531, 0.477, 0.018, -0.128, 0.395, 0.036, 0.353, 0.190],
        [0.194, 0.129, 0.326, 0.411, 0.552, 0.051, 0.310, 0.109, 0.159, 0.161, 0.134, 0.206, 0.454, 0.374, 0.307, 0.503, 0.297, 0.293, 0.005, 0.449, 0.428, 0.017, -0.096, 0.344, 0.048, 0.366, 0.210],
        [0.582, 0.615, 0.769, 0.864, 1.167, 0.029, 0.228, 0.063, 0.253, 0.505, 0.148, 0.275, 0.348, 0.307, 0.685, 0.822, 0.435, 0.454, 0.248, 0.499, 1.021, 0.009, 0.031, 0.475, 0.186, 0.897, 0.621],
        [0.997, 1.114, 1.344, 1.526, 2.093, 0.007, -0.052, 0.024, 0.245, 0.636, 0.093, 0.272, 0.597, 0.503, 0.822, 1.682, 0.911, 0.962, 0.290, 0.888, 1.472, -0.003, 0.073, 0.921, 0.319, 1.651, 1.177],
        [0.658, 0.763, 0.865, 0.970, 1.319, -0.005, -0.128, -0.007, 0.127, 0.372, 0.030, 0.132, 0.350, 0.297, 0.435, 0.911, 0.566, 0.621, 0.178, 0.559, 0.823, -0.004, 0.047, 0.600, 0.221, 1.062, 0.800],
        [0.856, 0.998, 1.098, 1.187, 1.692, -0.051, -0.286, -0.070, 0.115, 0.554, -0.023, 0.215, 0.340, 0.293, 0.454, 0.962, 0.621, 2.486, 0.338, 0.634, 1.030, -0.041, 0.355, 0.763, 0.291, 1.229, 1.007],
        [0.567, 0.712, 0.655, 0.684, 0.860, -0.052, -0.266, -0.102, 0.105, 0.506, -0.028, 0.062, -0.026, 0.005, 0.248, 0.290, 0.178, 0.338, 0.440, 0.204, 0.830, -0.022, 0.243, 0.300, 0.193, 0.722, 0.597],
        [0.699, 0.724, 0.943, 1.108, 1.334, 0.019, 0.052, 0.042, 0.225, 0.455, 0.103, 0.229, 0.531, 0.449, 0.499, 0.888, 0.559, 0.634, 0.204, 0.788, 0.896, 0.005, -0.004, 0.724, 0.226, 1.162, 0.822],
        [2.277, 2.930, 2.394, 2.416, 2.666, -0.030, 0.093, -0.047, 0.370, 1.390, 0.132, 0.374, 0.477, 0.428, 1.021, 1.472, 0.823, 1.030, 0.830, 0.896, 4.871, -0.021, 0.655, 1.253, 0.866, 2.668, 2.720],
        [-0.011, -0.021, -0.011, -0.010, -0.016, 0.018, 0.049, 0.024, 0.017, -0.016, 0.023, 0.010, 0.018, 0.017, 0.009, -0.003, -0.004, -0.041, -0.022, 0.005, -0.021, 0.015, -0.032, -0.004, 0.003, -0.002, -0.008],
        [0.371, 0.481, 0.361, 0.319, 0.424, -0.050, -0.150, -0.081, -0.039, 0.177, -0.068, -0.002, -0.128, -0.096, 0.031, 0.073, 0.047, 0.355, 0.243, -0.004, 0.655, -0.032, 1.545, 0.111, 0.136, 0.396, 0.468],
        [1.054, 1.314, 1.298, 1.465, 1.664, -0.018, -0.228, -0.041, 0.165, 0.565, 0.019, 0.149, 0.395, 0.344, 0.475, 0.921, 0.600, 0.763, 0.300, 0.724, 1.253, -0.004, 0.111, 1.330, 0.421, 1.651, 1.299],
        [0.779, 1.050, 0.786, 0.773, 0.816, -0.019, -0.211, -0.055, 0.053, 0.317, -0.023, 0.018, 0.036, 0.048, 0.186, 0.319, 0.221, 0.291, 0.193, 0.226, 0.866, 0.003, 0.136, 0.421, 0.768, 0.941, 1.010],
        [2.557, 2.927, 2.986, 3.328, 3.485, -0.067, -0.683, -0.176, 0.298, 1.178, -0.028, 0.128, 0.353, 0.366, 0.897, 1.651, 1.062, 1.229, 0.722, 1.162, 2.668, -0.002, 0.396, 1.651, 0.941, 8.475, 3.246],
        [2.660, 3.161, 2.682, 2.638, 2.776, -0.072, -0.683, -0.181, 0.157, 0.966, -0.082, 0.048, 0.190, 0.210, 0.621, 1.177, 0.800, 1.007, 0.597, 0.822, 2.720, -0.008, 0.468, 1.299, 1.010, 3.246, 7.916],
    ])
)

# Calculated from Ken French Data Library, 1927-2019
factor_data = dict(
    asset_classes = [
        "Mkt-RF",
        "HML",
        "Mom",
    ],
    means = [
        6.4,
        3.6,
        6.5,
    ],
    stdevs = [
        18.5,
        12.1,
        16.3,
    ],
    correlations = [
        [ 1             ],
        [ 0.23,  1      ],
        [-0.34, -0.41, 1],
    ]
)

# Calculated from Ken French Data Library and AQR data set, 1985-2019
factor_data_with_tsmom = dict(
    asset_classes = [
        "Mkt-RF",
        "HML",
        "Mom",
        "TSMOM^EQ",
    ],
    means = [
        7.7,
        1.3,
        5.1,
        14.0,
    ],
    stdevs = [
        15.1,
        10.0,
        15.7,
        12.3,
    ],
    correlations = [
        [ 1                   ],
        [-0.21,  1   ,        ],
        [-0.19, -0.20, 1      ],
        [-0.02, -0.13, 0.40, 1],
    ],
)

# From RAFI data above, but using just a few asset classes
mini_rafi_data = dict(
    asset_classes = [
        "US Market", "Global ex-US", "Commodities", "Intermediate Bonds"
    ],
    means  = [ 5,  5,  1, -1],
    stdevs = [16, 17, 16,  4],
    correlations = [
        [ 1                   ],
        [ 0.9,  1             ],
        [ 0.3,  0.4,  1       ],
        [-0.3, -0.3, -0.1,  1 ],
    ]
)

trend_overlay_data = dict(
    asset_classes = [
        # Market: global equities
        # Long Val/Mom: long-only value and momentum, like QVAL/QMOM/IVAL/IMOM
        # Trend Overlay: short the market when it is in a downtrend, otherwise
        #   do nothing (0% nominal return)
    "Market", "Long Val/Mom", "Trend Overlay"
    ],
    means  = [ 5,  8, -2],  # nominal, not real
    stdevs = [16, 16, 14],
    correlations = [
        [ 1   ,  0.84, -0.74],
        [ 0.84,  1   , -0.59],
        [-0.74, -0.59,  1   ],
    ]
)

my_favorite_data = dict(
    asset_classes = [
        # Market: global equities
        # Val/Mom: long-only value and momentum, like QVAL/QMOM/IVAL/IMOM
        # VMOT: VMOT
        # ManFut: managed futures, like AQR Time Series Momentum data set
        # GVAL: cheap stocks in cheap countries, like GVAL
        "Market", "Val/Mom", "VMOT", "ManFut", "GVAL"
    ],
    means  = [ 3,  6,  6,  3,  9],
    stdevs = [16, 16, 13, 15, 22],
    correlations = [
        [ 1                     ],
        [ 0.8,  1               ],
        [ 0.5,  0.8, 1          ],
        [ 0  ,  0  , 0.2, 1     ],
        [ 0.8,  0.8, 0.6, 0  , 1],
    ]
)


class Optimizer:
    def __init__(self, asset_data_to_use, short_cost=0, leverage_cost=0):
        '''
        short_cost and leverage_cost are percentages.
        '''
        self.short_cost = short_cost / 100
        self.leverage_cost = leverage_cost / 100

        self.asset_classes = asset_data_to_use['asset_classes']
        means = asset_data_to_use['means']
        stdevs = asset_data_to_use.get('stdevs')
        correlations = asset_data_to_use.get('correlations')
        covariances = asset_data_to_use.get('covariances')

        if covariances is None:
            covariances = [
                [correl * stdev1 * stdev2 / 100 for correl, stdev2 in zip(row, stdevs)]
                for row, stdev1 in zip(correlations, stdevs)
            ]

        if stdevs is None:
            # TODO: test this
            stdevs = [covariances[i][i] for i in range(len(means))]

        for i in range(len(covariances)):
            if len(covariances[i]) < len(covariances):
                for j in range(i + 1, len(covariances)):
                    covariances[i].append(covariances[j][i])

        # Convert from percentage to proportion
        self.means = np.array([x/100 for x in means])
        self.stdevs = np.array([x/100 for x in stdevs])
        self.covariances = np.array([[x/100 for x in row] for row in covariances])

        self.NO_CONSTRAINT = optimize.LinearConstraint(
            # This is a "constraint" that's not actually constraining. Use this if you
            # want to disable a particular constraint.
            [0 for _ in means],
            lb=0, ub=1
        )

    def borrowing_costs(self, weights, scale=1):
        total_short_cost = abs(sum(w for w in weights if w < 0)) * self.short_cost
        total_leverage_cost = max(0, sum(w for w in weights if w > 0) - scale) * self.leverage_cost
        return total_short_cost + total_leverage_cost

    def neg_return(self, weights):
        return -np.dot(weights, self.means + self.stdevs**2 / 2)

    def neg_return_with_uncertainty(self, weights):
        param_uncertainty = 0.2
        num_samples = 5000

        accum_return = 0
        if getattr(self, 'monte_carlo_means', None) is None:
            self.monte_carlo_means = []
            arithmetic_means = self.means + self.stdevs**2 / 2
            for i in range(num_samples):
                # for now, assume stdev and correlation are fixed
                self.monte_carlo_means.append(
                    np.array([np.random.normal(mean, param_uncertainty * mean)
                              for mean in arithmetic_means]))

        for i in range(num_samples):
            means = self.monte_carlo_means[i]
            accum_return += -np.dot(weights, means)

        return accum_return / num_samples

        # return -np.dot(weights, self.means + self.stdevs**2 / 2)

    def neg_sharpe(self, weights):
        ret = np.dot(weights, self.means + self.stdevs**2 / 2)
        stdev = np.sqrt(np.dot(np.dot(self.covariances, weights), weights))
        return -ret / stdev

    def geometric_mean(self, weights, extra_cost=0):
        '''Approximation of the geometric mean using the formula from
        Estrada (2010), "Geometric Mean Maximization: An Overlooked Portfolio Approach?"
        https://web.iese.edu/jestrada/PDF/Research/Refereed/GMM-Extended.pdf

        See also
        Bernstein & Wilkinson (1997), "Diversification, Rebalancing, and the Geometric Mean Frontier"
        https://www.effisols.com/basics/rebal.pdf
        '''
        arithmetic_means = self.means + self.stdevs**2 / 2
        arithmetic_mean = np.dot(weights, arithmetic_means) - extra_cost
        variance = np.dot(np.dot(self.covariances, weights), weights)
        return np.log(1 + arithmetic_mean) - variance / (2 * (1 + arithmetic_mean)**2)

    def geometric_mean_with_uncertainty(self, weights, extra_cost=0):
        '''Find expected geometric mean when parameter values are uncertain.'''
        param_uncertainty = 0.2
        num_samples = 5000

        accum_gmean = 0
        saved_means = self.means
        saved_stdevs = self.stdevs
        if getattr(self, 'monte_carlo_means', None) is None:
            self.monte_carlo_means = []
            self.monte_carlo_stdevs = []
            for i in range(num_samples):
                # for now, assume stdev and correlation are fixed
                self.monte_carlo_means.append(
                    np.array([np.random.normal(mean, param_uncertainty * mean)
                              for mean in saved_means]))
                self.monte_carlo_stdevs.append(
                    np.array([np.random.normal(stdev, param_uncertainty * stdev)
                              for stdev in saved_stdevs]))

        for i in range(num_samples):
            self.means = self.monte_carlo_means[i]
            self.stdevs = self.monte_carlo_stdevs[i]
            accum_gmean += self.geometric_mean(weights, extra_cost)

        self.means = saved_means
        self.stdevs = saved_stdevs
        return accum_gmean / num_samples

    def mvo(self, max_stdev=None, target_leverage=None, shorts_allowed=False):
        '''
        Find either the asset allocation with the highest return for a given
        standard deviation, or the allocation with the highest Sharpe ratio for
        a given amount of leverage.

        max_stdev should be provided as a percentage.
        '''
        assert(max_stdev is not None or target_leverage is not None)

        shorts_constraint = self.NO_CONSTRAINT
        variance_constraint = self.NO_CONSTRAINT

        if not shorts_allowed:
            shorts_constraint = optimize.LinearConstraint(
                np.identity(len(self.asset_classes)),  # identity matrix
                lb=[0 for _ in self.asset_classes],
                ub=[np.inf for _ in self.asset_classes]
            )

        if max_stdev is not None:
            max_stdev /= 100  # convert from percentage to proportion
            # You're not allowed to invest money in nothing for a guaranteed 0% return.
            # That would be ok if returns were nominal, but this program uses real
            # returns.
            leverage_constraint = optimize.LinearConstraint(
                [1 for _ in self.asset_classes],
                lb=1, ub=np.inf
            )
            variance_constraint = optimize.NonlinearConstraint(
                lambda weights: np.dot(np.dot(self.covariances, weights), weights),
                lb=0, ub=max_stdev**2
            )
            optimand = self.neg_return
        else:
            leverage_constraint = optimize.LinearConstraint(
                [1 for _ in self.asset_classes],
                lb=target_leverage, ub=target_leverage
            )
            optimand = self.neg_sharpe

        opt = optimize.minimize(
            optimand,
            x0=[0.01 for _ in self.asset_classes],
            constraints=[shorts_constraint, leverage_constraint, variance_constraint]
        )
        print("Return: {:.2f}%".format(100 * self.geometric_mean(opt.x)))
        print("Stdev: {:.2f}%".format(
            100 * np.sqrt(np.dot(np.dot(self.covariances, opt.x), opt.x))
        ))
        print()
        print("\n".join(["{}\t{:.3f}".format(name, weight)
                        for name, weight in zip(self.asset_classes, opt.x)]))


    def maximize_gmean_below_uncertainty(
            self,
            max_stdev=None,
            max_leverage=None,
            shorts_allowed=True,
            exogenous_portfolio_weight=0,
    ):
        '''
        Find the average portfolio over a Monte Carlo sample of optimal portfolios,
        rather than the optimal portfolio over a Monte Carlo sample.
        '''
        # TODO: refactor this function, it's almost exactly the same as `maximize_gmean`
        # If you short $1, you need to hold this much in cash, and you can invest the rest
        short_margin_requirement = 1.0 / 2
        endogenous_prop = 1 - exogenous_portfolio_weight

        leverage_constraint = optimize.LinearConstraint(
            [1 for _ in self.asset_classes],
            lb=endogenous_prop, ub=np.inf
        )
        variance_constraint = self.NO_CONSTRAINT
        shorts_constraint = self.NO_CONSTRAINT

        if not shorts_allowed:
            shorts_constraint = optimize.LinearConstraint(
                np.identity(len(self.asset_classes)),  # identity matrix
                lb=[0 for _ in self.asset_classes],
                ub=[np.inf for _ in self.asset_classes]
            )
        elif max_leverage is not None:
            shorts_constraint = optimize.NonlinearConstraint(
                lambda weights: sum([min(x, 0) for x in weights]),
                lb=-endogenous_prop, ub=0
            )

        if max_leverage is not None:
            overall_max_leverage = max_leverage * endogenous_prop
            leverage_constraint = optimize.NonlinearConstraint(
                lambda weights: sum([x if x > 0 else x * (1 - short_margin_requirement)
                                    for x in weights]),
                lb=endogenous_prop, ub=overall_max_leverage
            )

        if max_stdev is not None:
            max_stdev /= 100  # convert from percentage to proportion
            max_stdev *= endogenous_prop
            variance_constraint = optimize.NonlinearConstraint(
                lambda weights: np.dot(np.dot(self.covariances, weights), weights),
                lb=0, ub=max_stdev**2
            )

        def optimand(weights):
            weights2 = [w for w in weights]
            weights2[0] += exogenous_portfolio_weight
            return -self.geometric_mean(
                weights2, self.borrowing_costs(weights, scale=endogenous_prop))

        num_samples = 5000
        param_uncertainty = 0.2
        accum_weights = np.array([0.0 for _ in self.asset_classes])
        saved_means = self.means
        for i in range(num_samples):
            self.means = [np.random.normal(mean, param_uncertainty * mean) for mean in saved_means]
            opt = optimize.minimize(
                optimand,
                x0=[0.01 for _ in self.asset_classes],
                constraints=[leverage_constraint, shorts_constraint, variance_constraint]
            )
            accum_weights += opt.x

        optimal_weights = accum_weights / num_samples
        personal_weights = [weight / endogenous_prop for weight in optimal_weights]

        personal_stdev = np.sqrt(np.dot(np.dot(self.covariances, personal_weights), personal_weights))

        # see https://en.wikipedia.org/wiki/Covariance#Covariance_of_linear_combinations
        # Assumes first element of list is the market
        correl_with_market = (
            [w * self.covariances[i][0] for (i, w) in enumerate(personal_weights)]
            / (personal_stdev * np.sqrt(self.covariances[0][0]))
        )

        print("| Allocation | |")
        print("|-|-|")
        print("\n".join(["| {} | {:.0f}% |".format(name, 100 * weight / endogenous_prop)
                        for name, weight in zip(self.asset_classes, optimal_weights)]))
        print()
        print("| Summary Statistics | |")
        print("|-|-|")
        print("| Total Altruistic Return | {:.2f}% |".format(100 * -optimand(optimal_weights)))
        print("| Personal Return | {:.2f}% |".format(
            100 * self.geometric_mean(
                personal_weights, self.borrowing_costs(personal_weights, scale=1))))
        print("| Personal Standard Deviation | {:.2f}% |".format(
            100 * personal_stdev
        ))
        print("| Personal Correlation to Market | {:.2f} |".format(correl_with_market))
        print()

    def maximize_gmean(
            self,
            max_stdev=None,
            max_leverage=None,
            shorts_allowed=True,
            exogenous_portfolio_weight=0,
            exogenous_weights=[1, 0, 0, 0],
    ):
        '''max_stdev should be provided as a percentage.

        Unlike `mvo`, you may provide both a max_stdev and a max_leverage.
        '''
        # If you short $1, you need to hold this much in cash, and you can invest the rest
        short_margin_requirement = 1.0 / 2
        endogenous_prop = 1 - exogenous_portfolio_weight

        leverage_constraint = optimize.LinearConstraint(
            [1 for _ in self.asset_classes],
            lb=endogenous_prop, ub=np.inf
        )
        variance_constraint = self.NO_CONSTRAINT
        shorts_constraint = self.NO_CONSTRAINT

        if not shorts_allowed:
            shorts_constraint = optimize.LinearConstraint(
                np.identity(len(self.asset_classes)),  # identity matrix
                lb=[0 for _ in self.asset_classes],
                ub=[np.inf for _ in self.asset_classes]
            )
        elif max_leverage is not None:
            shorts_constraint = optimize.NonlinearConstraint(
                lambda weights: sum([min(x, 0) for x in weights]),
                lb=-endogenous_prop, ub=0
            )

        if max_leverage is not None:
            overall_max_leverage = max_leverage * endogenous_prop
            leverage_constraint = optimize.NonlinearConstraint(
                lambda weights: sum([x if x > 0 else x * (1 - short_margin_requirement)
                                    for x in weights]),
                lb=endogenous_prop, ub=overall_max_leverage
            )

        if max_stdev is not None:
            max_stdev /= 100  # convert from percentage to proportion
            max_stdev *= endogenous_prop
            variance_constraint = optimize.NonlinearConstraint(
                lambda weights: np.dot(np.dot(self.covariances, weights), weights),
                lb=0, ub=max_stdev**2
            )

        def optimand(weights):
            weights2 = [w for w in weights]
            for i in range(len(weights2)):
                weights2[i] += exogenous_portfolio_weight * exogenous_weights[i]
            return -self.geometric_mean(
                weights2, self.borrowing_costs(weights, scale=endogenous_prop))

        opt = optimize.minimize(
            optimand,
            x0=[0.01 for _ in self.asset_classes],
            constraints=[leverage_constraint, shorts_constraint, variance_constraint]
        )

        personal_weights = [weight / endogenous_prop for weight in opt.x]

        personal_stdev = np.sqrt(np.dot(np.dot(self.covariances, personal_weights), personal_weights))

        # see https://en.wikipedia.org/wiki/Covariance#Covariance_of_linear_combinations
        # Assumes first element of list is the market
        correl_with_market = (
            sum(wi * wj * self.covariances[i][j]
                for (i, wi) in enumerate(personal_weights)
                for (j, wj) in enumerate(exogenous_weights))
            / (personal_stdev * np.sqrt(np.dot(np.dot(self.covariances, exogenous_weights), exogenous_weights)))
        )

        print("| Allocation | |")
        print("|-|-|")
        print("\n".join(["| {} | {:.0f}% |".format(name, 100 * weight / endogenous_prop)
                        for name, weight in zip(self.asset_classes, opt.x)]))
        print()
        print("| Summary Statistics | |")
        print("|-|-|")
        print("| Total Altruistic Return | {:.2f}% |".format(100 * -optimand(opt.x)))
        print("| Personal Return | {:.2f}% |".format(
            100 * self.geometric_mean(
                personal_weights, self.borrowing_costs(personal_weights, scale=1))))
        print("| Personal Standard Deviation | {:.2f}% |".format(
            100 * personal_stdev
        ))
        print("| Personal Correlation to Market | {:.2f} |".format(correl_with_market))
        print()

Optimizer(
    my_favorite_data,
    short_cost=0.25,
    leverage_cost=1,
).maximize_gmean(
    max_stdev=20,
    exogenous_portfolio_weight=0.99,
    exogenous_weights=[1, 0, 0, 0, 0],
)
