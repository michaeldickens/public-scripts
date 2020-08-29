'''

metaculus-convergence.py
------------------------

Author: Michael Dickens
Created: 2020-08-19

'''

from collections import defaultdict
import json
import pprint


def load_questions():
    with open('db/continuous-convergence.json', 'r') as f:
        json_str = f.read()
        return json.loads(json_str)


def convergence_by_num_predictions(questions):
    '''
    Find average distance to true answer as a function of the number of predictions.

    TODO: I was assuming that continuous questions are normalized to between 0
    and 1, where 0 is the minimum possible answer and 1 is the maximum. But
    that is not always the case. It is not clear to me how the normalization
    works.

    '''
    sum_squared_error_by_num_predictions = defaultdict(int)
    count_by_num_predictions = defaultdict(int)
    mean_squared_error_by_num_predictions = {}

    for question in questions:
        for point_in_time in question['prediction_timeseries']:
            sum_squared_error_by_num_predictions[point_in_time['num_predictions']] += (
                (point_in_time['avg'] - question['resolution'])**2
            )
            count_by_num_predictions[point_in_time['num_predictions']] += 1

    mean_squared_error_by_num_predictions = {
        k: sum_squared_error_by_num_predictions[k] / count_by_num_predictions[k]
        for k in sum_squared_error_by_num_predictions
        # filter out any prediction counts with fewer than 30 data points
        if count_by_num_predictions[k] >= 30
    }

    return mean_squared_error_by_num_predictions


def convergence_by_time(questions):
    # TODO
    pass


questions = load_questions()
mse = average_convergence(questions)
print("\n".join(["{}\t{}".format(k, v) for (k, v) in sorted([(k, mse[k]) for k in mse], key=lambda x: x[0])]))
