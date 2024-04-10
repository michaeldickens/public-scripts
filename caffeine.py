"""

caffeine.py
-----------

Author: Michael Dickens <michael@mdickens.me>
Created: 2024-02-23

Script for analyzing my caffeine self-experiment.

"""

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import re
import sys
from scipy.stats import linregress, t, ttest_1samp, ttest_ind
from sklearn import linear_model
import numpy as np
import os


PLOT_FONT_SIZE = 14

with open("doc/reaction-times.org", "r") as f:
    lines = f.readlines()

header_line = lines.index("* TODO Reaction Time Test\n")
assert header_line >= 0
lines = lines[header_line:]

# find first line that starts with "** "
for i, line in enumerate(lines):
    if line.startswith("** "):
        break

lines = lines[i:]

# Each entry is formatted as:
# ** [datetime] reactiontime C? miscellaneous-notes
#    more-miscellaneous-notes
pattern = re.compile(
    r"\*\* \[(?P<date_time>\d{4}-\d{2}-\d{2} \w{3} \d{2}:\d{2})\] (?P<number>\d+)(?P<has_c> C)?(?P<text_after> .*)?"
)

reaction_data = []
for line in lines:
    m = pattern.match(line)
    if m:
        # Extract components
        datetime_obj = datetime.strptime(m.group("date_time"), "%Y-%m-%d %a %H:%M")
        number = int(m.group("number"))
        has_c = bool(m.group("has_c"))
        text_after = m.group("text_after")

        # Append extracted data to the list
        reaction_data.append(
            {
                "datetime": datetime_obj,
                "reaction_time": number,
                "C": has_c,
                "extra": text_after or "",
            }
        )
    elif line[:2] == "**" and line[3:9] != "ignore":
        print(f"Line did not match pattern: {line}")


nocaf_data = [d for d in reaction_data if not d["C"]]
caf_data = [d for d in reaction_data if d["C"]]


def get_daily_RTs(data):
    daily_RTs = {}
    for d in data:
        date = d["datetime"].date()
        if date not in daily_RTs:
            daily_RTs[date] = []
        daily_RTs[date].append(d["reaction_time"])
    return daily_RTs


def get_daily_averages(daily_RTs, phase_range):
    # calculate average reaction time per day
    daily_averages = []
    for times in sorted(daily_RTs.items(), key=lambda x: x[0]):
        if (
            times[0] >= phase_range[0]
            and times[0] <= phase_range[1]
        ):
            daily_averages.append((times[0], np.mean(times[1])))
    return daily_averages


nocaf_daily_RTs = get_daily_RTs(nocaf_data)
caf_daily_RTs = get_daily_RTs(caf_data)

FULL_RANGE = (datetime(2024, 1, 30).date(), datetime(2024, 3, 29).date())
PHASE_1_RANGE = (datetime(2024, 1, 30).date(), datetime(2024, 2, 23).date())
PHASE_2_RANGE = (datetime(2024, 2, 24).date(), datetime(2024, 3, 3).date())
# PHASE_3_RANGE = (datetime(2024, 3, 4).date(), datetime(2024, 3, 29).date())
PHASE_3_RANGE = (datetime(2024, 3, 5).date(), datetime(2024, 3, 29).date())  # excluding the day where I accidentally clicked too early

phase_ranges = {
    "calibration": PHASE_1_RANGE,
    "abstinence": PHASE_2_RANGE,
    "experimental": PHASE_3_RANGE,
}

caf_deltas = []
for date in caf_daily_RTs.keys() & nocaf_daily_RTs.keys():
    if date < PHASE_1_RANGE[0] or date > PHASE_1_RANGE[1]:
        continue
    caf_delta = np.mean(caf_daily_RTs[date]) - np.mean(nocaf_daily_RTs[date])
    caf_deltas.append(caf_delta)

BASELINE_BENEFIT = np.mean(caf_deltas)


def pair_stdev():
    """Calculate standard deviation of pairs of reaction times."""
    prev_entry = None
    diffs = []
    stdevs = []
    for d in reaction_data:
        if prev_entry is None:
            prev_entry = d
        elif (d["datetime"] - prev_entry["datetime"]).seconds < 3 * 60:
            diffs.append(abs(d["reaction_time"] - prev_entry["reaction_time"]))
            stdevs.append(np.std([d["reaction_time"], prev_entry["reaction_time"]]))
            prev_entry = None
        else:
            prev_entry = d

    print(f"Average pair difference: {np.mean(diffs):.1f} ms, average pair stdev: {np.mean(stdevs):.1f} ms")


def run_regression(daily_RTs, phase_range, phase_name, null_slope=0, verbose=True):
    daily_averages = get_daily_averages(daily_RTs, phase_range)

    # run a regression on average reaction time per day
    x = [(d[0] - daily_averages[0][0]).days for d in daily_averages]
    y = [d[1] for d in daily_averages]

    # the null hypothesis is that the slope equals null_slope
    y = y - np.array(x) * null_slope
    slope, intercept, r_value, p_value, stderr = linregress(x, y)

    if verbose:
        print(f"\nRegression over {phase_name}:\n\tslope = {slope:.2f} ms/day, p-value = {p_value:.3f}\n")
    return slope, intercept, stderr


def caf_vs_nocaf(reaction_data, phase_range):
    """
    Compare caffeine vs. no-caffeine reaction times and print summary
    statistics.
    """
    reaction_data = [d for d in reaction_data if d["datetime"].date() >= phase_range[0] and d["datetime"].date() <= phase_range[1]]
    caf_days = set([d["datetime"].date() for d in reaction_data if d["C"]])

    caf_diffs = []
    for day in caf_days:
        caf_times = [d["reaction_time"] for d in reaction_data if d["C"] and d["datetime"].date() == day]
        nocaf_times = [d["reaction_time"] for d in reaction_data if not d["C"] and d["datetime"].date() == day]
        if len(caf_times) > 0 and len(nocaf_times) > 0:
            caf_diffs.append(np.mean(caf_times) - np.mean(nocaf_times))

    mean_diff = np.mean(caf_diffs)
    # t-test
    t_stat, p_value = ttest_1samp(caf_diffs, 0)
    print(f"\nCaffeine vs. no caffeine on same day:\n\tmean difference = {mean_diff:.1f} ms, t-stat = {t_stat:.1f}, p-value = {p_value:.3f}\n")


def first_vs_second(reaction_data, phase_range):
    """
    Determine if reaction time is better/worse on the first trial of the
    day vs. the second trial of the day. (Any trials that occur within a few
    minutes of each other are counted as part of the same trial.)
    """
    reaction_data = [d for d in reaction_data if d["datetime"].date() >= phase_range[0] and d["datetime"].date() <= phase_range[1]]
    all_days = set([d["datetime"].date() for d in reaction_data])
    caf_days = set([d["datetime"].date() for d in reaction_data if d["C"]])
    nocaf_days = all_days - caf_days

    diffs = []
    for day in nocaf_days:
        data = [d for d in reaction_data if d["datetime"].date() == day]
        # put data into groups of 2
        data = [data[i:i+2] for i in range(0, len(data), 2)]
        if len(data) != 2:
            print(f"Warning: {len(data)} test set{'s' if len(data) > 1 else ''} on {day}, expected 2")
        if len(data) < 2:
            continue
        avg_times = [np.mean([d["reaction_time"] for d in pair]) for pair in data]
        diffs.append(avg_times[1] - avg_times[0])

    mean_diff = np.mean(diffs)
    # t-test
    t_stat, p_value = ttest_1samp(diffs, 0)
    print(f"\nFirst vs. second test on non-caffeine days:\n\tmean difference = {mean_diff:.1f} ms, t-stat = {t_stat:.1f}, p-value = {p_value:.3f}\n")


def plot_likelihood(daily_RTs, phase_range, improvement_direction, slope_adjustment=0):
    """
    Create a plot of the likelihood function for caffeine retention as
    estimated by the given reaction times.
    """
    domain = (0, 2)
    num_x_points = (domain[1] - domain[0]) * 100 + 1
    sample_size = len([d for d in daily_RTs if d >= phase_range[0] and d <= phase_range[1]])
    phase_length = (phase_range[1] - phase_range[0]).days

    # Convert a slope to a total change in reaction time, and then normalize it
    # such that a change in reaction time of BASELINE_BENEFIT corresponds to a
    # retention of 0. improvement_direction should be -1 if a decreasing RT
    # represents an improvement and +1 if an increasing score represents an
    # improvement (such as during abstinence phase).
    scale = improvement_direction * BASELINE_BENEFIT / phase_length
    likelihoods = np.zeros(num_x_points)
    slope, _, stderr = run_regression(daily_RTs, phase_range, "", null_slope=slope_adjustment, verbose=False)
    for i, slope_param in enumerate(np.linspace((domain[0] - 1) * scale, (domain[1] - 1) * scale, num_x_points)):
        # This gives the likelihood of observing `slope_param` when the true
        # slope is `slope`, whereas we want the likelihood of observing `slope`
        # when the true slope is `slope_param`. But the two numbers are equal
        # so that's ok.
        likelihoods[i] = t.pdf((slope - slope_param) / stderr, sample_size - 1)

    # find the sum of likelihoods where x < 1
    sum_likelihoods = np.sum(likelihoods)
    sum_likelihoods_1 = np.sum(likelihoods[:int(num_x_points * (1 - domain[0]) / (domain[1] - domain[0]))])
    print(f"Probability of likelihoods < 1: {sum_likelihoods_1 / sum_likelihoods:.3f}")
    print(f"Likelihood at 1: {likelihoods[101]:.3f}")
    print(f"CDF at slope = 0: {t.cdf(slope / stderr, sample_size - 1)}")

    # plot likelihood function
    plt.figure()
    plt.rc('font', size=PLOT_FONT_SIZE)
    plt.plot(np.linspace(domain[0], domain[1], num_x_points), likelihoods)
    plt.xlabel("Caffeine retention")
    plt.ylabel("Likelihood")
    plt.title("Likelihood function of caffeine retention")
    fig = plt.gcf()
    fig.set_size_inches(1.25 * fig.get_size_inches(), forward=True)
    return likelihoods


def abstinence_phase_bayesian_update():
    """Apply a Bayesian update on the evidence of caffeine effectiveness based
    on the abstinence phase, using a uniform prior. An effectiveness of 0 (full
    habituation) predicts a slope of -13/9 ms/day (calibration phase showed
    caffeine improves reaction time by 13ms, and abstinence phase lasted 9 days).
    An effectiveness of 1 (no habituation) predicts a slope of 0.
    """
    likelihoods = plot_likelihood(nocaf_daily_RTs, PHASE_2_RANGE, 1.04)
    # likelihoods = plot_likelihood(nocaf_daily_RTs, PHASE_2_RANGE, 0)

    prior = np.ones(len(likelihoods)) / len(likelihoods)
    prior *= likelihoods
    prior /= np.sum(prior)
    print(f"Posterior mean: {np.sum(prior * np.linspace(0, 1, 100)):.2f}")

    plt.figure()
    plt.plot(np.linspace(0, 1, 100), prior * 100)
    plt.ylim(0)
    plt.xlabel("Caffeine effectiveness")
    plt.ylabel("Probability density")
    plt.title("Posterior distribution of caffeine effectiveness")
    plt.show()


def plot_caf_or_nocaf(daily_RTs, phase_range, name, color):
    """
    Create a scatterplot and regression line for a set of reaction times,
    but do not render the plot yet. Return the slope of the regression and
    standard error of the reaction times.
    """
    slope, intercept, stderr = run_regression(daily_RTs, phase_range, "experimental phase", verbose=False)
    daily_averages = get_daily_averages(daily_RTs, phase_range)
    x = [(d[0] - daily_averages[0][0]).days for d in daily_averages]
    y = [d[1] for d in daily_averages]
    plt.rc('font', size=PLOT_FONT_SIZE)
    plt.plot(x, [slope * xi + intercept for xi in x], label=f"{name} regression line (slope = {slope:.2f})", color=color)
    plt.plot(x, y, "o", label=f"{name} daily averages", color=color)
    return slope, stderr


def read_sleep_data(filename='data.csv'):
    # TODO: this is copy/pasted from sync_scripts/sleep.py because I don't know
    # how to get relative imports to work
    sleep_data = {}
    with open(filename, 'r') as f:
        header = None
        for line in f:
            if header is None:
                header = line.lower().split(';')
                continue

            entries = line.split(';')

            wakeup_index = header.index('end')
            wakeup_date = datetime.strptime(entries[wakeup_index], "%Y-%m-%d %H:%M:%S").date()

            sleep_quality_index = header.index('sleep quality')
            quality = int(entries[sleep_quality_index][:-1])  # cut off the % sign

            try:
                time_in_bed_index = header.index('time in bed')
                hours_in_bed, minutes_in_bed = entries[time_in_bed_index].split(':')
                time_in_bed = int(hours_in_bed) * 60 + int(minutes_in_bed)
            except ValueError:
                time_in_bed_index = header.index('time in bed (seconds)')
                time_in_bed = float(entries[time_in_bed_index]) / 60

            try:
                time_asleep_index = header.index('time asleep (seconds)')
                time_asleep = float(entries[time_asleep_index]) / 60
            except ValueError:
                time_asleep = Noneo

            today_data = {
                'quality': quality,
                'minutes': time_in_bed,
                'minutes_asleep': time_asleep,
            }
            if wakeup_date in sleep_data:
                # Sometimes I have an extra short entry if I restart Sleep
                # Cycle. As a heuristic, ignore whichever entry is shorter.
                if sleep_data[wakeup_date]['minutes'] > today_data['minutes']:
                    today_data = sleep_data[wakeup_date]

            sleep_data[wakeup_date] = today_data

    return sleep_data


def control_for_sleep():
    """
    Create a plot comparing raw reaction times vs. reaction times
    controlling for time spent in bed the previous night.

    Note to readers: This function won't work because it tries to open a file
    on my computer containing the sleep data, and that path doesn't exist on
    your computer.
    """
    sleep_data = read_sleep_data("/home/mdickens/programs/sync_scripts/sleep/data-2021-2024.csv")
    full_date_range = (PHASE_1_RANGE[0], PHASE_3_RANGE[1])
    phase_name = "experimental"
    sleep_key = "minutes"

    sleep_durations = []
    reaction_times = []
    for date_index in range((full_date_range[1] - full_date_range[0]).days + 1):
        date = full_date_range[0] + timedelta(days=date_index)
        if date not in sleep_data:
            print("Sleep data missing date:", date)
            continue
        if date not in nocaf_daily_RTs:
            print("RTs missing date:", date)
            continue
        minutes_asleep = sleep_data[date][sleep_key]
        nocaf_RT = np.mean(nocaf_daily_RTs[date])
        sleep_durations.append(minutes_asleep)
        reaction_times.append(nocaf_RT)

    paired_data = np.array(list(zip(sleep_durations, reaction_times)))
    slope, intercept, r_value, p_value, stderr = linregress(paired_data)
    print(f"\nRegression over sleep data:\n\tslope = {60*slope:.2f} ms/hour, p-value = {p_value:.5f}\n")

    phase_range = phase_ranges[phase_name]
    daily_RTs = caf_daily_RTs
    uncontrolled_reaction_times = {date: daily_RTs[date] for date in sorted(daily_RTs.keys()) if date >= phase_range[0] and date <= phase_range[1] and date in daily_RTs}

    # control reaction times for sleep duration
    avg_sleep_duration = np.mean([sleep_data[date][sleep_key] for date in uncontrolled_reaction_times.keys()])
    avg_RT = np.mean([np.mean(RTs) for RTs in uncontrolled_reaction_times.values()])
    controlled_reaction_times = {
        date: RT + avg_RT - (intercept + slope * sleep_data[date][sleep_key])
        for date, RT in uncontrolled_reaction_times.items()
    }

    # plot_likelihood(controlled_reaction_times, phase_range, 1)
    # fig = plt.gcf()
    # fig.set_size_inches(1.25 * fig.get_size_inches(), forward=True)
    # plt.savefig(f"doc/caf-{phase_name}-likelihood-controlled.png", dpi=125)
    # plt.show()

    print(f"Average sleep: {avg_sleep_duration / 60:.2f} hours. r^2 = {r_value**2:.2f}")
    plot_caf_or_nocaf(uncontrolled_reaction_times, phase_range, "raw", "blue")
    plot_caf_or_nocaf(controlled_reaction_times, phase_range, "controlled for sleep", "forestgreen")
    plt.xlabel("Days since start of phase")
    plt.ylabel("Reaction time (ms)")
    plt.title(f"Reaction time during {phase_name} phase")
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(1.25 * fig.get_size_inches(), forward=True)
    res = plt.savefig(f"doc/caf-{phase_name}-regression-controlled.png", dpi=125)
    plt.show()


def plot_regression():
    """
    Create a plot of a regression of reaction time across days for caffeine
    and no-caffeine trials.
    """
    phase_name = "experimental"
    phase_range = phase_ranges[phase_name]

    plt.figure()
    plot_caf_or_nocaf(nocaf_daily_RTs, phase_range, "nocaf", "blue")
    plot_caf_or_nocaf(caf_daily_RTs, phase_range, "caf", "orangered")

    plt.xlabel("Days since start of phase")
    plt.ylabel("Reaction time (ms)")
    plt.title(f"Reaction time during {phase_name} phase")
    plt.legend()

    # expand plot width
    fig = plt.gcf()
    fig.set_size_inches(1.25 * fig.get_size_inches(), forward=True)

    # save figure
    plt.savefig(f"doc/caf-{phase_name}-regression-skip-day-0.png", dpi=125)
    plt.show()
