"""
simon_ftorek_vs_mc.py
---------------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2024-03-19

Generate Monte Carlo samples for the caffeine effectiveness likelihood function
and compare the result to the Simon-Ftorek approximation to see how accurate
the approximation is.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import erf

def ratio_dist_pdf_simon_ftorek(z, mu_x, sigma_x, mu_y, sigma_y):
    """Approximation for the PDF of a ratio of two independent noncentral
    normal distributions. From Wikipedia:
    https://en.wikipedia.org/wiki/Ratio_distribution#Normal_ratio_distributions
    """
    p = mu_x / (np.sqrt(2) * sigma_x)
    q = mu_y / (np.sqrt(2) * sigma_y)
    r = mu_x / mu_y

    pdf_approximation = (
        (1 / np.sqrt(np.pi))
        * (p / erf(q))
        * (1 / r)
        * (1 + (p**2 / q**2) * (z / r))
        / ((1 + (p**2 / q**2) * ((z / r) ** 2)) ** (3 / 2))
        * np.exp(
            -((p**2 * ((z / r) - 1) ** 2) / (1 + (p**2 / q**2) * ((z / r) ** 2)))
        )
    )

    return pdf_approximation

domain = (0, 1)

# generate 10,000 samples from norm(437, 6.5), norm(420, 7.9), and norm(417, 5.7)
lopla = np.random.normal(437, 6.5, 10000)
locaf = np.random.normal(420, 7.9, 10000)
hicaf = np.random.normal(417, 5.7, 10000)

numer = hicaf - lopla
denom = locaf - lopla
ratio = numer / denom

# baseline benefit must be at least +1 standard error
RESTRICT_TO_1SE = True

if RESTRICT_TO_1SE:
    condition = (ratio > domain[0]) & (ratio < domain[1]) & (-denom >= np.sqrt(7.9**2 + 6.5**2))
else:
    condition = (ratio > domain[0]) & (ratio < domain[1])

numer = numer[condition]
denom = denom[condition]
ratio = ratio[condition]

numer_mean = (417 - 437)
numer_std = np.sqrt((5.7**2 + 6.5**2))
denom_mean = (420 - 437)
denom_std = np.sqrt((7.9**2 + 6.5**2))

sf_domain = np.linspace(domain[0], domain[1], 1000)

if RESTRICT_TO_1SE:
    # Use empirical mean/SE instead of true mean/SE because cutting off the distribution at +1SE will shift the mean/SE
    sf_approximation = ratio_dist_pdf_simon_ftorek(sf_domain, np.mean(numer), np.std(numer), np.mean(denom), np.std(denom))
else:
    sf_approximation = ratio_dist_pdf_simon_ftorek(sf_domain, numer_mean, numer_std, denom_mean, denom_std)

# normalize to integrate to 1. technically doesn't matter because SF is
# approximating a likelihood function not a probability distribution, but this
# makes the plot look nicer
sf_approximation /= np.trapz(sf_approximation, sf_domain)

mc_mean = np.mean(ratio)
sf_mean = np.trapz(sf_domain * sf_approximation, sf_domain)

sns.histplot(ratio, kde=True, stat="density", bins=100, label=f"Monte Carlo ({mc_mean:.2f})")
sns.lineplot(x=sf_domain, y=sf_approximation, lw=2, label=f"Simon-Ftorek approximation ({sf_mean:.2f})", color="red")

plt.title("Caffeine effectiveness likelihood function")
plt.xlabel("Retention")
plt.ylabel("Likelihood")
plt.legend()

plt.savefig("doc/caf-MC-vs-SF.png", dpi=150)
plt.show()
