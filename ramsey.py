'''

ramsey.py
---------

Author: Michael Dickens
Created: 2020-06-24

'''

import math

from scipy.special import gamma, gammaincc


def consumption_with_declining_discount(alpha, delta, r, eta):
    consumption_coef =  (
        ((delta + r*eta - r) / eta)**(1 - alpha/eta)
        # TODO: unclear if the gamma(1 - alpha/eta) term should be there,
        # Python's definition of gammaincc is weird
        / (gamma(1 - alpha/eta) * gammaincc(1 - alpha/eta, (delta + r*eta - r)/eta))
    )
    def consumption(t):
        return (
            consumption_coef
            * (1 + t)**(-alpha/eta)
            # ((delta + r*eta - r)/eta) / math.exp((r - delta)/eta)
            * math.exp((r - delta)/eta * (1 + t))
        )

    prev_capital = 1

    for i in range(1, 1 + 1000000):
        t = i / 10000
        dt = t / i
        consum = consumption(t)
        capital = prev_capital * (1 + dt * r) - dt * consum
        capital_growth_rate = (capital - prev_capital) / prev_capital / dt
        consumption_rate = consum / capital
        consumption_growth_rate = (consum - consumption(t-dt)) / consum / dt
        # TODO: This starts to become inaccurate over time, I think due to floating point rounding issues
        if i % int(100/dt) == 2 or (i % int(10/dt) == 2 and i*dt < 100):
            print("{}\t{}\t{}\t{}\t{}".format(t, capital, capital_growth_rate, consumption_rate, consumption_growth_rate))

        prev_capital = capital


consumption_with_declining_discount(alpha=0.01, delta=0.001, r=0.05, eta=1)
