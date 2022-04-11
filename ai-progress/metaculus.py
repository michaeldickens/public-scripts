"""

meticulous.py
------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2022-03-21

"""

from datetime import datetime
import json
import numpy as np
import requests

def fetch_question_json(question_num):
    '''
    For locally-saved questions, load the question from a file. Otherwise, send
    an HTTP request to the Metaculus API.
    '''
    if question_num == 3479:
        with open('data/metaculus-3479.json') as fp:
            return json.loads(fp.read())
    else:
        response = requests.get('https://WWW.metaculus.com/api2/questions/{}/'.format(question_num))
        return json.loads(response.content)


def denormalize(data, point):
    '''
    See https://github.com/oughtinc/ergo/blob/master/ergo/scale.py
    '''
    log_base = data['possibilities']['scale']['deriv_ratio']
    if log_base == 1:
        # linear
        return (point + 1) / 200
    else:
        years = 1 + 180 * (log_base**(point / 200) - 1) / (log_base - 1)

        # assumes linear progress in some sense. eg could be 'progress' is a
        # logarithmic function, and the thing being measured moves
        # exponentially.
        # growth_rate = 1 / years
        growth_rate = np.log(1 + 1 / years)
        return growth_rate


def prediction_vol(data):
    predictions_normalized = list(enumerate(data['community_prediction']['full']['y']))
    total = sum(h for (x, h) in predictions_normalized)
    predictions = [(denormalize(data, x), h / total) for (x, h) in predictions_normalized]
    mean = sum(x * h for (x, h) in predictions)
    stdev = np.sqrt(sum(h * (x - mean)**2 for (x, h) in predictions))
    print("Point-in-time prediction: mean {:.3f}, stdev {:.3f}".format(mean, stdev))


def vol_over_time(data):
    # drop the first 50 b/c not enough predictions yet
    limited_data = data['prediction_timeseries'][50:]
    time_series_normalized = [entry['distribution']['x0'] for entry in limited_data]  # mean
    # time_series_normalized = [entry['community_prediction']['q2'] for entry in limited_data]  # median
    time_series = [denormalize(data, 200 * x) for x in time_series_normalized]
    total_time_gap = limited_data[-1]['t'] - limited_data[0]['t']
    time_gap = total_time_gap / len(limited_data)
    time_gap_years = time_gap / (60 * 60 * 24 * 365)
    mean = np.mean(time_series)
    stdev = np.std(time_series)
    print("Time series median: mean {:.3f}, stdev {:.3f}, stdev per year {:.3f}".format(
        mean, stdev, stdev / np.sqrt(time_gap_years)))


data = fetch_question_json(3479)
prediction_vol(data)
vol_over_time(data)
