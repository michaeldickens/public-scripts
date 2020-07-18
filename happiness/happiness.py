import csv
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def read_csv(filename: str) -> Dict[str, List[Optional[float]]]:
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)

        header = None
        csv_data = defaultdict(list)
        for row in reader:
            if header is None:
                header = row
                continue

            csv_data[header[0]].append(row[0])
            for i in range(1, len(row)):
                csv_data[header[i]].append(float(row[i]))

    return csv_data


def fit_happiness(csv_data):
    log_xs = np.array(csv_data['log GDP per capita']).reshape(-1, 1)
    log_ys = np.log(csv_data['standard error of happiness'])
    # log_ys = np.array(csv_data['happiness'])

    regression = LinearRegression()
    regression.fit(log_xs, log_ys)
    predicted_ys = regression.predict(log_xs)
    print("y = {:.2f} x + {:.2f} (r^2 {:.2f}, MSE {:.2f})".format(
        regression.coef_[0], regression.intercept_,
        r2_score(log_ys, predicted_ys), mean_squared_error(np.exp(log_ys), np.exp(predicted_ys))))

fit_happiness(read_csv("country-happiness.csv"))
