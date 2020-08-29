'''

manage-db.py
------------

Author: Michael Dickens
Created: 2020-08-19

metaculus.json taken from https://niplav.github.io/range_and_forecasting_accuracy.html

'''

from datetime import datetime
import json

def to_epoch(timestamp):
    try:
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
    except ValueError:
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def continuous_question_convergence():
    '''
    Filter down to just the required data for determining continuous question convergence.
    '''

    with open('db/metaculus.json', 'r') as f:
        json_str = f.read()
        pages = json.loads(json_str)

    output_json = []

    for page in pages:
        for question in page['results']:
            if question['possibilities']['type'] != 'continuous':
                continue
            if question['resolution'] is None:
                continue

            prediction_timeseries_output = []

            for point_in_time in question['prediction_timeseries']:
                distribution = point_in_time['distribution']
                prediction_timeseries_output.append({
                    't': point_in_time['t'],
                    'stdev': distribution['s' if 's' in distribution else 'stdev'],
                    'avg': distribution['x0' if 'x0' in distribution else 'avg'],

                    # TODO: unclear if I should use distribution['num'] or point_in_time['num_predictions']
                    # 'num_predictions': distribution['num'],
                    'num_predictions': point_in_time['num_predictions'],
                })

            output_json.append({
                'id': question['id'],
                'resolution': question['resolution'],
                'publish_time': to_epoch(question['publish_time']),
                'close_time': to_epoch(question['close_time']),
                'resolve_time': to_epoch(question['resolve_time']),
                'prediction_timeseries': prediction_timeseries_output,
            })

    with open('db/continuous-convergence.json', 'w') as f:
        output_str = json.dumps(output_json)
        f.write(output_str)


continuous_question_convergence()
