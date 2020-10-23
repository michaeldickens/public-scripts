"""

mvo.py
------

Author: Michael Dickens <mdickens93@gmail.com>
Created: 2020-10-05

"""

from pprint import pprint

import numpy as np
from scipy import optimize


# Returns and covariances downloaded from Research Affiliates' Asset Allocation
# Interactive on 2020-10-05.
#
# I originally tried to do this on the .xslx that I downloaded from RAFI, but
# LibreOffice's solver apparently sucks at solving optimization problems.


rafi_asset_classes = [
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
    "REITS",
    "United States Cash",
    "US Commercial Real Estate",
    "Global DM Ex-US Long/Short Equity",
    "US Long/Short Equity",
    "Europe LBO",
    "US LBO",
]

# Some asset classes are commented out and set to -10 so they won't be used
rafi_means = np.array([
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
    -10,  # 0.7,  # Bank Loans
    -0.8,  # Global Ex-US Corporates
    1.5,  # REITS
    -0.6,  # United States Cash
    -10,  # 2.3,  # US Commercial Real Estate
    -10,  # 3.2,  # Global DM Ex-US Long/Short Equity
    -10,  # 1.0,  # US Long/Short Equity
    -10,  # 6.8,  # Europe LBO
    -10,  # 0.0,  # US LBO
])

rafi_covariances = np.array([
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

factor_asset_classes = [
    "Mkt-RF",
    "HML",
    "Mom",
]

# 1927-2019, calculated from Ken French Data Library
factor_means = [
    6.4,
    3.6,
    6.5,
]

factor_covariances = [
    [ 3.423,  0.515, -1.025],
    [ 0.515,  1.464, -0.809],
    [-1.025, -0.809,  2.657],
]

factor_asset_classes_with_tsmom = [
    "Mkt-RF",
    "HML",
    "Mom",
    "TSMOM^EQ",
]

factor_means_with_tsmom = [
    7.7,
    1.3,
    5.1,
    15.8,
]

factor_covariances_with_tsmom = [
    [2.28, -0.32, -0.45, -0.08],
    [-0.32, 1.00, -0.31, -0.35],
    [-0.45, -0.31, 2.47, 1.69],
    [-0.08, -0.35, 1.69, 7.24],
]

# asset_classes = rafi_asset_classes
# means = rafi_means
# covariances = rafi_covariances

asset_classes = [
    "60/40", "AQR TSMOM"
]

# from AQR's "A Century of Evidence of Trend-Following Investing"
covariances = None
means = [7.8, 11.2]
stdevs = [10.8, 9.7]
correlations = [
    [1, 0],  # correlation to equities is 0.00, to bonds is -0.04
    [0, 1]
]

def neg_return(weights):
    return -np.dot(weights, means)

def neg_sharpe(weights):
    ret = np.dot(weights, means)
    stdev = np.sqrt(np.dot(np.dot(covariances, weights), weights))
    return -ret / stdev

def mvo(max_stdev=None, target_leverage=None):
    '''
    max_stdev should be provided as a percentage
    '''
    global covariances
    assert(max_stdev is not None or target_leverage is not None)

    if covariances is None:
        covariances = [
            [correl * stdev1 * stdev2 / 100 for correl, stdev2 in zip(row, stdevs)]
            for row, stdev1 in zip(correlations, stdevs)
        ]

    no_shorts_constraint = optimize.LinearConstraint(
        np.identity(len(means)),  # identity matrix
        lb=[0 for _ in means],
        ub=[np.inf for _ in means]
    )

    if max_stdev is not None:
        # You're not allowed to invest money in nothing for a guaranteed 0% return.
        # That would be ok if returns were nominal, but this program uses real
        # returns.
        leverage_constraint = optimize.LinearConstraint(
            [1 for _ in means],
            lb=1, ub=np.inf
        )
        variance_constraint = optimize.NonlinearConstraint(
            lambda weights: np.dot(np.dot(covariances, weights), weights),
            lb=0, ub=max_stdev**2 / 100
        )
        optimand = neg_return
    else:
        leverage_constraint = optimize.LinearConstraint(
            [1 for _ in means],
            lb=target_leverage, ub=target_leverage
        )
        variance_constraint = optimize.LinearConstraint(
            # not an actual constraint
            [0 for _ in means],
            lb=0, ub=1
        )
        optimand = neg_sharpe

    opt = optimize.minimize(
        optimand,
        x0=[0.01 for _ in means],
        constraints=[no_shorts_constraint, leverage_constraint, variance_constraint]
    )
    print("Return: {:.2f}%".format(np.dot(opt.x, means)))
    print("Stdev: {:.2f}%".format(
        np.sqrt(np.dot(np.dot(covariances, opt.x), opt.x) / 100) * 100
    ))
    print()
    print("\n".join([name + "\t" + (str(x) if x > 1e-10 else "0") for name, x in zip(asset_classes, opt.x)]))

mvo(target_leverage=1)
