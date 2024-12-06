"""

protest_outcomes.py
-------------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2024-12-04

Code for my protest outcomes meta-analysis.

"""

from collections import namedtuple
import numpy as np

"""
Madestam et al. (2013):

(with election and demographic controls)

TODO: what are election controls?

- rain decreases Republican party House vote share by 1.55% (stderr 0.69). n = 2758
- 1% of population protesting increases Republican vote share by 18.81% (stderr 7.85)
"""

"""
Teeselink and Melios (2021):

model 2 (with protest and demographic controls)

TODO: what are protest controls?

- 1% rain chance decreases Democratic presidential vote share by 2.5% (stderr 1.2). n = 3053
- 1% of population protesting increases Democratic vote share by 2.4% (stderr 0.7)

(model 3 got similar results)
"""

class Outcome(namedtuple("Outcome", ["mean", "stderr", "n"])):
    def stdev(self):
        return self.stderr * np.sqrt(self.n)

    def variance(self):
        return self.stderr ** 2 * self.n


def pooled_outcome(outcomes):
    """
    Calculate pooled outcome using a random-effects model.
    """
    means = [outcome.mean for outcome in outcomes]
    overall_mean = np.sum([outcome.mean * outcome.n for outcome in outcomes]) / np.sum([outcome.n for outcome in outcomes])
    between_study_variance = np.var(means)

    # idk
    # see https://en.wikipedia.org/wiki/Random_effects_model
