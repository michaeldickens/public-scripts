"""

eff_api.py
----------

Author: Michael Dickens <michael@mdickens.me>
Created: 2022-03-25

Manage EFF's AI progress data set.

https://www.eff.org/ai/metrics

"""

import json
import math
import numpy as np

from datetime import datetime

SECONDS_PER_YEAR = 60 * 60 * 24 * 365


with open("data/eff-ai-progress.json") as fp:
    data = json.loads(fp.read())

metrics = {
    metric['measures'][0]['metric']: [
        [
            datetime.strptime(dic['date'], "%Y-%m-%d"),
            dic['value']
        ]
        for dic in metric['measures']
        if dic['value'] > 0
    ]
    for problems in data['problems'] for metric in problems['metrics']
    if len(metric['measures']) >= 2
}

metrics = {k: metric for k, metric in metrics.items() if len(metric) > 0}

score_growth = {}
for k, metric in metrics.items():
    # Assert that dates are non-descending
    for i in range(1, len(metric)):
        assert metric[i][0] >= metric[i-1][0], "{} < {}".format(metric[i][0], metric[i-1][0])

    net_ascending = 0
    value_type = 'proportion'
    for i in range(1, len(metric)):
        if metric[i][1] > 100:
            value_type = 'big number'
        elif metric[i][1] > 1 and value_type != 'big number':
            value_type = 'percentage'
        if metric[i][1] > metric[i-1][1]:
            net_ascending += 1
        elif metric[i][1] < metric[i-1][1]:
            net_ascending -= 1

    # Convert all measures to descending
    if net_ascending > 0:
        for i in range(len(metric)):
            metric[i][1] = {
                'proportion': 1 - metric[i][1],
                'percentage': (100 - metric[i][1]) / 100,
                'big number': 1 / metric[i][1],
            }[value_type]

    # Convert dates to years
    for i in range(len(metric)):
        metric[i][0] = metric[i][0].year

    # Remove any measures that are not better than an older measure
    prev = metric[0][1]
    for i in range(1, len(metric)):
        if metric[i][1] >= prev:
            metric[i] = None
        else:
            prev = metric[i][1]
    metric = [x for x in metric if x is not None]

    # If multiple measures have the same year, keep only the first one
    prev = metric[0][0]
    for i in range(1, len(metric)):
        if metric[i][0] == prev:
            metric[i] = None
        else:
            prev = metric[i][0]
    metric = [x for x in metric if x is not None]

    # Convert scores to log score growth
    metric_growth = {
        metric[i][0]: np.log((metric[i-1][1] / metric[i][1])**(1.0 / (metric[i][0] - metric[i-1][0])))
        for i in range(1, len(metric))
    }

    if len(metric_growth) > 0:
        score_growth[k] = metric_growth


min_year = min(min(metric.keys()) for k, metric in score_growth.items())
max_year = max(max(metric.keys()) for k, metric in score_growth.items())


# Aggregate growth per year across all metrics
avg_growth_per_year = {}
sample_size_per_year = {}
for year in range(min_year, max_year + 1):
    growth_for_year = [
        growth[year]
        for growth in score_growth.values()
        if year in growth
    ]
    sample_size_per_year[year] = len(growth_for_year)
    if len(growth_for_year) == 0:
        # If a year has no entries, that means no model on the list had an improvement that year
        avg_growth_per_year[year] = 0
    else:
        avg_growth_per_year[year] = np.mean(growth_for_year)


def correlate_to_semiconductors()::
    # Returns for MSCI Semiconductors and Semiconductor Equipment Index
    # Transcribed from https://www.msci.com/documents/10199/43fbeb9e-05ce-4b94-8d98-4ab50d9b5409
    semiconductor_ret = {y: np.log(1 + x/100) for y, x in {
        2008: -49.61,
        2009: 61.79,
        2010: 10.34,
        2011: -4.06,
        2012: 0.31,
        2013: 36.47,
        2014: 28.48,
        2015: -2.8,
        2016: 30.45,
        2017: 40.38,
        2018: -12.85,
        2019: 54.54,
        2020: 46.70,
        2021: 51.98,
    }.items()}

    # Transcribed from https://www.slickcharts.com/sp500/returns
    spy_ret = {y: np.log(1 + x/100) for y, x in {
        2008: -37.00,
        2009: 26.46,
        2010: 15.06,
        2011: 2.11,
        2012: 16.00,
        2013: 32.39,
        2014: 13.69,
        2015: 1.38,
        2016: 11.96,
        2017: 21.83,
        2018: -4.38,
        2019: 31.49,
        2020: 18.40,
        2021: 28.71,
    }.items()}

    print([semiconductor_ret[y] - spy_ret[y] for y in range(2012, 2022)])

    pairs = []
    for y in avg_growth_per_year.keys() & semiconductor_ret.keys():
        pairs.append((avg_growth_per_year[y], semiconductor_ret[y]-  spy_ret[y]))

    print(np.corrcoef([x[0] for x in pairs], [x[1] for x in pairs]))


# print(avg_growth_per_year)
# print(np.mean([x for x in avg_growth_per_year.values() if not math.isnan(x)]))
# print(np.std([x for x in avg_growth_per_year.values() if not math.isnan(x)]))
