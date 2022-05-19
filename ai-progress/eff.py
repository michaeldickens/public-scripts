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

    # Measures are not all on the same scale. Use some heuristics to guess
    # which scale is being used
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

    # If multiple measures occur in the same year, keep only the first one
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
# min_year = 2012
max_year = max(max(metric.keys()) for k, metric in score_growth.items())


# Aggregate growth per year across all metrics
avg_growth_per_year = {}
sample_size_per_year = {}
for year in range(min_year, max_year + 1):
    growth_in_year = [
        growth[year]
        for growth in score_growth.values()
        if year in growth
    ]
    sample_size_per_year[year] = len(growth_in_year)
    if len(growth_in_year) == 0:
        # If a year has no entries, that means no model on the list had an improvement that year
        avg_growth_per_year[year] = 0
    else:
        avg_growth_per_year[year] = np.mean(growth_in_year)


def correlate_to_semiconductors():
    def convert_rets(rets):
        return {y: np.log(1 + x/100) for y, x in rets.items()}

    # Returns for MSCI Semiconductors and Semiconductor Equipment Index
    # Transcribed from https://www.msci.com/documents/10199/43fbeb9e-05ce-4b94-8d98-4ab50d9b5409
    semiconductor_ret = convert_rets({
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
    })

    # Transcribed from https://www.slickcharts.com/sp500/returns
    spy_ret = convert_rets({
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
    })

    # Global market portfolio, calculated using StockStrategy/French
    gmp_ret = convert_rets({
        2008: -37.82,
        2009: 32.98,
        2010: 25.62,
        2011: 3.94,
        2012: 25.49,
        2013: 31.58,
        2014: 9.28,
        2015: 3.22,
        2016: 5.17,
        2017: 35.89,
        2018: -16.41,
        2019: 13.84,
    })

    # Simulated version of Advanced Alpha Architect, calculated using StockStrategy/French
    aaa_ret = convert_rets({
        2008: -34.01,
        2009: 29.47,
        2010: 31.96,
        2011: -14.61,
        2012: 12.73,
        2013: 65.65,
        2014: 5.87,
        2015: 0.53,
        2016: 11.28,
        2017: 27.67,
        2018: -17.51,
        2019: 7.51,
    })

    pairs = []
    # for y in avg_growth_per_year.keys() & semiconductor_ret.keys():
        # pairs.append((avg_growth_per_year[y], semiconductor_ret[y]))
    for y in avg_growth_per_year.keys() & spy_ret.keys():
        pairs.append((avg_growth_per_year[y], spy_ret[y]))
    # for y in avg_growth_per_year.keys() & gmp_ret.keys():
        # pairs.append((avg_growth_per_year[y], gmp_ret[y]))
    # for y in avg_growth_per_year.keys() & aaa_ret.keys():
    #     pairs.append((avg_growth_per_year[y], aaa_ret[y]))

    print(np.corrcoef([x[0] for x in pairs], [x[1] for x in pairs]))


correlate_to_semiconductors()
# print(avg_growth_per_year)
# print(np.mean([x for x in avg_growth_per_year.values() if not math.isnan(x)]))
# print(np.std([x for x in avg_growth_per_year.values() if not math.isnan(x)]))
