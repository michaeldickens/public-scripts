
"""

metaculus.py
------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2022-03-21

"""

from datetime import datetime
import json
import numpy as np
import requests

NUM_PREDICTION_SEGMENTS = 200


def fetch_question_json(question_num, cache=True):
    '''
    For locally-saved questions, load the question from a file. Otherwise, send
    an HTTP request to the Metaculus API.
    '''
    downloaded_questions = [3479, 5121]
    if cache and question_num in downloaded_questions:
        with open('data/metaculus-{}.json'.format(question_num)) as fp:
            return json.loads(fp.read())
    else:
        response = requests.get('https://WWW.metaculus.com/api2/questions/{}/'.format(question_num))
        return json.loads(response.content)


def denormalize(data, point):
    '''
    Denormalization algorithm taken from
    https://github.com/oughtinc/ergo/blob/master/ergo/scale.py
    '''
    log_base = data['possibilities']['scale']['deriv_ratio']
    min_year = datetime.strptime(data['possibilities']['scale']['min'], "%Y-%m-%d").year
    max_year = datetime.strptime(data['possibilities']['scale']['max'], "%Y-%m-%d").year
    num_years = max_year - min_year + 1
    if log_base == 1:
        # linear scale
        years = num_years * (point + 1) / NUM_PREDICTION_SEGMENTS
        return 1 / years
    else:
        # log scale
        years = 1 + num_years * (log_base**(point / NUM_PREDICTION_SEGMENTS) - 1) / (log_base - 1)

        # assumes linear progress in some sense. eg could be 'progress' is a
        # logarithmic function, and the thing being measured moves
        # exponentially.
        growth_rate = 1 / years
        # growth_rate = np.log(1 + 1 / years)
        return growth_rate


def prediction_vol(data):
    '''
    Calculate the volatility across user predictions.
    '''
    predictions_normalized = list(enumerate(data['community_prediction']['full']['y']))
    total = sum(density for (x, density) in predictions_normalized)
    predictions = [(denormalize(data, x), density / total) for (x, density) in predictions_normalized]
    mean = sum(x * density for (x, density) in predictions)
    stdev = np.sqrt(sum(density * (x - mean)**2 for (x, density) in predictions))
    print("Point-in-time community prediction: mean {:.3f}, stdev {:.3f}".format(mean, stdev))


def vol_over_time(data):
    '''
    Calculate the volatility of the community prediction over time.
    '''
    # drop the first 50 b/c not enough predictions yet
    limited_data = data['prediction_timeseries'][50:]
    # time_series_normalized = [entry['distribution']['x0'] for entry in limited_data]  # mean
    time_series_normalized = [entry['community_prediction']['q2'] for entry in limited_data]  # median
    time_series = [denormalize(data, NUM_PREDICTION_SEGMENTS * x) for x in time_series_normalized]
    total_time_gap = limited_data[-1]['t'] - limited_data[0]['t']
    time_gap = total_time_gap / len(limited_data)
    time_gap_years = time_gap / (60 * 60 * 24 * 365)
    mean = np.mean(time_series)
    stdev = np.std(time_series)
    print("Time series median prediction: mean {:.3f}, stdev {:.3f}, stdev per year {:.3f}".format(
        mean, stdev, stdev / np.sqrt(time_gap_years)))


data = fetch_question_json(5121, cache=False)
prediction_vol(data)
vol_over_time(data)
