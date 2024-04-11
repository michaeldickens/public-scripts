"""

paper.py
--------

Author: Michael Dickens <michael@mdickens.me>
Created: 2024-03-06

Statistical analysis for a literature review on caffeine habituation.
Citations for the papers used are listed along with the data.

"""

import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.special import erf
from scipy.stats import beta, cauchy, norm, t

import squigglepy as sq  # requires numeric-methods branch


def from_triple(triple):
    """Take a triple of [LoPla, LoCaf, HiCaf] and return [HiCaf - LoPla, LoCaf - LoPla]."""
    return [
        triple[2][0] - triple[0][0],
        np.sqrt(triple[2][1]**2 + triple[0][1]**2),
        triple[1][0] - triple[0][0],
        np.sqrt(triple[1][1]**2 + triple[0][1]**2),
    ]


global_x_values = np.linspace(0, 1, 101)


big_metrics_table = {
    # Rogers, P. J., Heatherley, S. V., Mullings, E. L., & Smith, J. E. (2013). Faster but not smarter: effects of caffeine and caffeine withdrawal on alertness and performance. Psychopharmacology, 226(2), 229-240. https://sci-hub.hkvisa.net/10.1007/s00213-012-2889-4
    # numbers are [retention mean, retention SE, benefit mean, benefit SE]
    "Rogers": {
        "SRT": [-19.929, 8.6372, -16.90, 10.241],
        "CRT": [-17.697, 10.4729, -11.80, 10.924],
        "memory": [0.45, 1.10, 0.67, 1.10],
        "tapping": [4.353, 2.8255, 5.99, 3.002],
    },

    # Haskell, C. F., Kennedy, D. O., Wesnes, K. A., & Scholey, A. B. (2005). Cognitive and mood improvements of caffeine in habitual consumers and habitual non-consumers of caffeine. Psychopharmacology, 179(4), 813-825. https://sci-hub.hkvisa.net/10.1007/s00213-004-2104-3
    "Haskell": {
        "SRT": [-17.64, 10.07, -5.6, 7.97],
        "DVRT": [-15.5, 7.12, -16.2, 7.16],
        "RVIP": [-0.5, 0.59, -0.21, 0.74],
        "spatial mem": [0.01, 0.02, 0.01, 0.02],
        "numeric mem RT": [-11.92, 15.84, -42.42, 15.91],
    },

    "Smith": {
        "FA speed": from_triple([
            (26.1, 3.16),
            (17.5, 3.37),
            (21.9, 3.16),
        ]),
        "CSRT": from_triple([
            (505.2, 8.85),
            (488.8, 0.35),
            (499.7, 7.43),
        ]),
        "SRT": from_triple([
            (364.2, 9.23),
            (322.8, 8.21),
            (326.1, 8.21),
        ]),
        "vigilance": from_triple([
            (27.15, 1.18),
            (30.47, 1.23),
            (29.28, 1.13),
        ]),
    },

    # Paul Hewlett & Andrew Smith (2007). Effects of repeated doses of caffeine on performance and alertness: new data and secondary analyses. , 22(6), 339–350. https://sci-hub.hkvisa.net/10.1002/hup.854
    #
    # See Table 5. Using Lo = non-consumer, Hi = higher consumer, Caf = 2 mg/kg. Restricted to metrics where caffeine had a noticeable effect on the non-consumer group.
    "Hewlett": {
        "FA speed": from_triple([
            (16.2, 4.8),
            (11.6, 4.2),
            (18.6, 5.7),
        ]),
        "SRT": from_triple([
            (336.3, 10.9),
            (326.9, 9.5),
            (323.8, 13.4),
        ]),
        "VR correct": from_triple([
            (92.0, 1.7),
            (93.0, 1.5),
            (91.0, 2.1),
        ]),
    },

    # Mood metrics from various studies
    "mood": {
        "sleepiness (Rogers)": from_triple([
            (2.793, 0.1922),
            (2.080, 0.1533),
            (2.512, 0.1728),
        ]),
        "alertness (Hewlett)": from_triple([
            (200.1, 16.4),
            (234.9, 14.1),
            (239.2, 20.0),
        ]),
        "alertness (Smith)": from_triple([
            (222.1, 8.82),
            (234.9, 7.06),
            (246.1, 6.47),
        ]),
    },

    # Beaumont, R., Cordery, P., Funnell, M., Mears, S., & James, L. (2017). Chronic ingestion of a low dose of caffeine induces tolerance to the performance benefits of caffeine. Journal of Sports Sciences, 35(19), 1920-1927.
    "Beaumont": {
        # numerator is Postcaf - Testpla for caffeine group, denominator is Precaf
        # - Testpla for caffeine group. placebo group is irrelevant except to
        # confirm that time is not a confounding factor (placebo group did not
        # perform better on Postcaf than Precaf)
        "kJ": [13.1, 18.2, 38.4, 19.9],
        "VO2": [
            2.34 - 2.31, np.sqrt(0.04**2 + 0.06**2),
            2.34 - 2.31, np.sqrt(0.04**2 + 0.04**2),
        ],
    },
}

# Lara, Beatriz; Ruiz-Moreno, Carlos; Salinero, Juan José; Del Coso, Juan; Sandbakk, Øyvind  (2019). Time course of tolerance to the performance benefits of caffeine. doi:10.1371/journal.pone.0210275
#
# I don't have the exact numbers. I extracted the numbers from graphs by using Gimp to count the height in pixels of each point. `pixels` is the number of pixels on the y-axis from 0 to 1.
lara_metrics = {
    "Wmax": {
        "means": [138, 143, 75, 59, 74, 81, 57, 76],
        "CIs": [126, 116, 100, 130, 101, 131, 102, 137],
        "pixels": 78,
    },
    "VO2max": {
        "means": [161, 91, 52, 53, 43, 31, 54, 31],
        "CIs": [120, 116, 102, 117, 133, 175, 116, 114],
        "pixels": 72,
    },
    "Wingate peak": {
        "means": [128, 118, 53, 87, 35, 88, 92, 82],
        "CIs": [173, 189, 219, 208, 129, 178, 125, 135],
        "pixels": 81,
    },
    "Wingate mean": {
        "means": [107, 72, 14, 21, 14, 52, 51, 40],
        "CIs": [124, 153, 146, 208, 174, 149, 133, 113],
        "pixels": 100,
    },

}
lara_Wmax_means = [138, 143, 75, 59, 74, 81, 57, 76]
lara_Wmax_CIs = [126, 116, 100, 130, 101, 131, 102, 137]
lara_Wmax_pixels = 78
lara_VO2max_means = [161, 91, 52, 53, 43, 31, 54, 31]
lara_VO2max_CIs = [120, 116, 102, 117, 133, 175, 116, 114]
lara_VO2max_pixels = 72
lara_Wingate_peak_means = [128, 118, 53, 87, 35, 88, 92, 82]
lara_Wingate_peak_CIs = [173, 189, 219, 208, 129, 178, 125, 135]
lara_Wingate_peak_pixels = 81
lara_Wingate_mean_means = [107, 72, 14, 21, 14, 52, 51, 40]
lara_Wingate_mean_CIs = [124, 153, 146, 208, 174, 149, 133, 113]
lara_Wingate_mean_pixels = 100


def ratio_dist_pdf_simon_ftorek(z, mu_x, sigma_x, mu_y, sigma_y):
    """Approximation for the PDF of a ratio of two independent noncentral
    normal distributions. From Wikipedia:
    https://en.wikipedia.org/wiki/Ratio_distribution#Normal_ratio_distributions

    This will produce an approximate distribution that is narrower than the true
    distribution. The true distribution can have undefined mean and infinite variance.
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


def ratio_dist_pdf_cauchy(x, mu_x, sigma_x, mu_y, sigma_y):
    """PDF of a ratio of two independent noncentral normal distributions,
    approximated as a Cauchy distribution. The ratio of two central normal
    distributions is a Cauchy distribution. The ratio of two noncentral normal
    distributions is *NOT* Cauchy, but this function naively assumes that it
    is. The advantage of this approximation is that it correctly represents the
    ratio distribution as being very wide.
    """
    return cauchy.pdf(x, mu_x / mu_y, sigma_x / sigma_y)


def ratio_dist_pdf_numeric(x, mu_x, sigma_x, mu_y, sigma_y):
    """PDF of a ratio of two independent noncentral normal distributions,
    calculated using Squigglepy on the analytic-numeric branch."""
    ratio_dist = sq.numeric(sq.norm(mean=mu_x, sd=sigma_x) / sq.norm(mean=mu_y, sd=sigma_y))
    return ratio_dist.pdf(x)


def bayes_update_on_ratio(prior_x, prior_p, m1, s1, m2, s2):
    """Perform a Bayes update on a ratio of distributions on the assumption
    that the distributions are independent and normally distributed. Warning:
    The approximation for the likelihood distribution is (potentially very)
    incorrect.
    """
    likelihood = ratio_dist_pdf_simon_ftorek(prior_x, m1, s1, m2, s2)
    posterior_p = prior_p * likelihood
    posterior_p *= 1 / np.sum(posterior_p)
    return posterior_p


def bayes_update_on_ratio_numeric(prior_x, prior_p, m1, s1, m2, s2):
    """Bayes update by numerically integrating over the ratio distribution,
    calculated using Squigglepy on the analytic-numeric branch. See
    https://github.com/rethinkpriorities/squigglepy/pull/61
    """
    ratio_dist = sq.numeric(sq.norm(mean=m1, sd=s1) / sq.norm(mean=m2, sd=s2))
    cdfs = ratio_dist.cdf(np.concatenate((prior_x, [prior_x[1] * len(prior_x)])))
    probs = np.diff(cdfs) / (prior_x[1] - prior_x[0])
    posterior_p = prior_p * probs
    posterior_p *= 1 / np.sum(posterior_p)
    return posterior_p


def update_on_metrics(prior, metrics, x_values):
    # plot prior and each of the posteriors, updating on the observed metrics one at a time
    if prior is None:
        keys = list(metrics.keys())
        prior = ratio_dist_pdf_simon_ftorek(x_values, *metrics[keys[0]])
        metrics = {k: metrics[k] for k in keys[1:]}
    posterior = prior
    for k in metrics:
        row = metrics[k]
        posterior = bayes_update_on_ratio(x_values, posterior, *row)
        plt.plot(x_values, posterior, label=k)
        # show a mark where the mean of the posterior is, with the same color as the curve, and put it on the curve
        mean = np.sum(posterior * x_values)
        plt.scatter(
            mean,
            posterior[np.argmin(abs(x_values - mean))],
            color=plt.gca().lines[-1].get_color(),
        )

    return posterior


def lara_regression(means, CIs, pixels, verbose=False):
    """Perform a weighted least squares regression on the data from Lara et
    al. Then impute baseline_benefit and habituated_benefit as the projected
    value of the regression on day 0 and day 20, respectively.
    """
    actual_retentions = global_x_values
    actual_slopes = (actual_retentions - 1) / 20

    # CI = 90% confidence interval
    means = np.array(means) / pixels
    CIs = np.array(CIs) / pixels
    days = np.array([1, 4, 6, 8, 13, 15, 18, 20])
    sample_size = 11
    point_tstat = t.ppf(0.95, sample_size - 1)
    std_errors = (CIs / 2) / point_tstat
    weights = 1 / std_errors**2

    # Weighted Least Squares model
    wls_model = sm.WLS(means, sm.add_constant(days), weights=weights)
    model = wls_model.fit()

    baseline_benefit = model.params[0]
    habituated_benefit = model.params[0] + 20 * model.params[1]
    midpoint_benefit = model.params[0] + 10 * model.params[1]
    retention = habituated_benefit / baseline_benefit

    if verbose:
        print(
            f"slope = {model.params[1] / baseline_benefit:.2f}, (p = {model.pvalues[1]:.2f})"
        )
        print("")
        print(f"Baseline benefit  : {baseline_benefit:.2f}")
        print(f"Habituated benefit: {habituated_benefit:.2f}")
        print(f"Retention         : {retention:.2f}")

    slope_stderr = model.bse[1]
    likelihood = t.pdf(
        actual_slopes, sample_size - 2, model.params[1] / baseline_benefit, slope_stderr
    )
    # normalize to integrate to 1
    likelihood /= np.trapz(likelihood, actual_slopes)
    return likelihood


def plot_posteriors():
    # prior distribution is a list of 200 segments from 0 to 2 as the area under the curve of a t distribution
    # prior = np.array([t.pdf(x, df=2, loc=0.5, scale=0.5) for x in np.linspace(0, 2, 200)])
    x_values = global_x_values
    prior = np.array([0.005 for x in x_values])
    prior *= 1 / np.sum(prior)
    prior_mean = np.sum(prior * x_values)
    plt.axvline(
        prior_mean,
        linestyle="--",
        label=f"prior = {prior_mean:.2f}",
    )
    plt.plot(x_values, prior)
    plt.scatter(
        prior_mean,
        prior[np.argmin(abs(x_values - prior_mean))],
        color=plt.gca().lines[-1].get_color(),
    )

    posterior = prior
    # posterior = update_on_metrics(posterior, haskell_metrics, x_values)
    # posterior = update_on_metrics(posterior, rogers_metrics, x_values)

    # joint likelihood
    # TODO: graph is inaccurate
    posterior = update_on_metrics(None, rogers_metrics, x_values)

    # plot mean on x axis
    posterior_mean = np.sum(posterior * x_values)
    plt.axvline(
        posterior_mean,
        color=plt.gca().lines[-1].get_color(),
        linestyle="--",
        label=f"posterior = {posterior_mean:.2f}",
    )

    plt.legend()
    plt.show()


def plot_joint_likelihood(study_name, metrics=None, weights=None, plot=True, save=True, colors=None, plot_approximations=False):
    """
    study_name: A keyword that names a study. If the study is one of the keys
        in `big_metrics_table`, then the metrics from the table are used.

    metrics: A dictionary where values are either:
        1. a 4-element list representing [mu_x, sigma_x, mu_y, sigma_y] where x = habituated benefit and y = baseline benefit; or
        2. a 101-element list representing a likelihood function

    weights: List of weights for each metric in the joint likelihood. Default
        weight is 1. Can be used to down-weight multiple metrics that come
        from the same study.

    plot: If True, draw plots of the likelihood functions.

    save: If True, save plots to files.

    colors: If given, override the default colors used in the plots.

    plot_approximations: If True, plot the likelihood functions of the
        three approximation methods for a single metric. If False, plot the likelihood functions of the metrics.
    """
    if metrics is None:
        metrics = big_metrics_table[study_name]
    elif study_name in big_metrics_table:
        metrics = {**big_metrics_table[study_name], **metrics}

    x_values = global_x_values
    labels = []
    likelihoods = []
    integrals = []
    if weights is None:
        weights = [1 for _ in metrics]
    assert len(weights) == len(metrics), f"{study_name}: {len(weights)} weights for {len(metrics)} metrics"
    plt.figure()

    if plot_approximations:
        funcs = [
            ratio_dist_pdf_simon_ftorek,
            ratio_dist_pdf_cauchy,
            ratio_dist_pdf_numeric,
        ]
        for func in funcs:
            likelihood = func(x_values, *metrics[list(metrics.keys())[0]])
            labels.append(func.__name__[len("ratio_dist_pdf_"):])
            likelihoods.append(likelihood)
    else:
        for k in metrics:
            row = metrics[k]

            # TODO: this is brittle duck typing
            if len(row) == 4:
                likelihood = ratio_dist_pdf_simon_ftorek(x_values, *row)
            else:
                likelihood = row
            labels.append(k)
            likelihoods.append(likelihood)  # append before normalizing

    # normalize the integral to 1. this effectively converts a likelihood
    # into a posterior probability over a uniform prior. but mainly I'm
    # doing it to make the graph look better
    likelihoods = [xs * len(x_values) / np.sum(xs) for xs in likelihoods]

    joint_likelihood = np.prod([l**w for l, w in zip(likelihoods, weights)], axis=0)
    joint_likelihood *= len(x_values) / np.sum(joint_likelihood)

    # plot joint likelihood in bold
    joint_mean = np.sum(joint_likelihood * x_values) / np.sum(joint_likelihood)
    if plot and len(metrics) > 1 and not plot_approximations:
        plt.plot(x_values, joint_likelihood, label=f"joint likelihood ({joint_mean:.2f})", linewidth=2)
        plt.scatter(
            joint_mean,
            joint_likelihood[np.argmin(abs(x_values - joint_mean))],
            color=plt.gca().lines[-1].get_color(),
        )

    if plot:
        for label, likelihood in zip(labels, likelihoods):
            likelihood_mean = np.sum(likelihood * x_values) / np.sum(likelihood)
            plt.plot(x_values, likelihood, label=f"{label} ({likelihood_mean:.2f})", alpha=0.5, color=colors.pop(0) if colors else None)
            plt.scatter(
                likelihood_mean,
                likelihood[np.argmin(abs(x_values - likelihood_mean))],
                color=plt.gca().lines[-1].get_color(),
                alpha=0.5,
            )

        plt.title(f"Likelihood functions of caffeine effectiveness ({study_name})")
        plt.xlabel("Retention")
        plt.ylabel("Likelihood")
        plt.gca().set_ylim(bottom=0)
        plt.legend()
        if save:
            plt.savefig(f"doc/caf-likelihood-{study_name.replace(' ', '-').replace(',', '')}.png", dpi=150)
        else:
            plt.show()

    return joint_likelihood


joint_likelihoods_all = {name: plot_joint_likelihood(name, weights=[1/len(big_metrics_table[name]) for _ in big_metrics_table[name]]) for name in ["Rogers", "Hewlett", "Haskell", "Smith", "Beaumont"]}
joint_likelihoods_cognitive = {name: joint_likelihoods_all[name] for name in ["Rogers", "Haskell", "Hewlett"]}

lara_likelihoods =  {
    k: lara_regression(**lara_metrics[k])
    for k in lara_metrics
}

joint_likelihoods_all["Lara"] = plot_joint_likelihood("Lara", metrics=lara_likelihoods, weights=[1/4 for _ in range(4)])

plot_joint_likelihood("mood")

joint_likelihoods_exercise = {
    "Lara": joint_likelihoods_all["Lara"],
    "Beaumont": joint_likelihoods_all["Beaumont"],
}
plot_joint_likelihood("joint exercise", metrics=joint_likelihoods_exercise)

joint_all = plot_joint_likelihood("joint cognition", metrics=joint_likelihoods_cognitive, plot=True)
plot_joint_likelihood("joint, all metrics", metrics=joint_likelihoods_all, colors=["g", "c", "#2ca02c", "m", "r", "#ff7f0e"])

prior = np.array([beta.pdf(x, 1, 1.5) for x in global_x_values])
prior *= 1 / np.sum(prior)
prior_mean = np.sum(prior * global_x_values)
posterior = prior * joint_all
posterior_mean = np.sum(posterior * global_x_values)
joint_mean = np.sum(joint_all * global_x_values) / 100
# plot posterior
plt.figure()
plt.plot(global_x_values, posterior, label=f"posterior ({posterior_mean:.2f})", linewidth=2)
plt.plot(global_x_values, prior, label=f"prior ({prior_mean:.2f})", linestyle="--")
plt.plot(global_x_values, joint_all / np.sum(joint_all), label=f"likelihood ({joint_mean:.2f})", linestyle="--")
plt.scatter(
    np.sum(posterior * global_x_values),
    posterior[np.argmin(abs(global_x_values - np.sum(posterior * global_x_values)))]
)
plt.legend()
plt.show()
