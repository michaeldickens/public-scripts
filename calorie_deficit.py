"""

calorie_deficit.py
------------------

Author: Michael Dickens
Created: 2025-04-22

Energy deficiency impairs resistance training gains in lean mass but not
strength: A meta-analysis and meta-regression.
https://onlinelibrary.wiley.com/doi/full/10.1111/sms.14075


"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

E = 1  # experimental (energy deficit) group
C = 0  # control group

# From Table S2 under Supporting Information. The table includes both lean mass
# and strength outcomes but this table only includes lean mass.
all_outcomes = [
    (E, 451, -0.06),
    (C, 51, 0.24),
    (E, 373, -0.26),
    (C, 67, 0.22),
    (E, 1012, 0.08),
    (C, -25, 0.35),
    (E, 405, -0.10),
    (C, -26, 0.02),
    (E, 124, 0.12),
    (C, 23, 0.09),
    (E, 259, -0.20),
    (C, 169, 0.07),
    (E, 1697, -0.05),
    (C, 79, 0.16),
    (E, 290, -0.07),
    (C, 34, 0.15),
    (E, 598, -0.29),
    (C, 72, 0.01),
    (E, 792, -0.13),
    (C, 37, 0.05),
    (E, 894, -0.18),
    (C, 7, 0.03),
    (E, 278, 0.07),
    (C, 144, 0.07),
    (E, 303, -0.30),
    (C, 169, 0.26),
    (E, 425, -0.34),
    (C, 67, 0.07),
    (E, 849, -0.29),
    (C, -34, 0.21),
    (E, 1190, -0.28),
    (C, 13, 0.15),
    (E, 665, -0.46),
    (C, 84, 0.32),
    (E, 719, -0.13),
    (C, 56, 0.26),
    (E, 892, -0.32),
    (C, 67, 0.18),
    (E, 459, -0.11),
    (C, 553, 0.61),
    (E, 194, 0.39),
    (C, 72, 0.39),
    (E, 438, 0.07),
    (C, 170, 0.63),
]


def save_plot(ed_only):
    if ed_only:
        outcomes = [outcome for outcome in all_outcomes if outcome[0] == E]
    else:
        outcomes = all_outcomes

    x = np.array([outcome[1] for outcome in outcomes])
    y = np.array([outcome[2] for outcome in outcomes])
    X = sm.add_constant(x/1000)  # divide by 1000 to make slope readable
    model = sm.OLS(y, X)
    results = model.fit()
    if ed_only:
        print("*** RT+ED Regression ***\n")
    else:
        print("*** RT+ED and RT+CON Regression ***\n")
    print(results.summary())
    print()

    x_ed = np.array([outcome[1] for outcome in all_outcomes if outcome[0] == E])
    x_con = np.array([outcome[1] for outcome in all_outcomes if outcome[0] == C])
    y_ed = np.array([outcome[2] for outcome in all_outcomes if outcome[0] == E])
    y_con = np.array([outcome[2] for outcome in all_outcomes if outcome[0] == C])

    plt.figure()
    plt.scatter(x_ed, y_ed, label="RT+ED")

    if not ed_only:
        plt.scatter(x_con, y_con, label="RT+CON")

    plt.plot(x, results.predict(X))
    plt.xlabel("Energy Deficit (kcal/day)", fontsize=14)
    plt.ylabel("Change in Lean Mass (effect size)", fontsize=14)
    plt.title("Effect of Energy Deficit on Lean Mass", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid()
    if ed_only:
        # save to file
        plt.savefig("/home/mdickens/programs/website/assets/images/RT+ED.png")
    else:
        plt.savefig(
            "/home/mdickens/programs/website/assets/images/RT+ED-and-RT+CON.png"
        )


save_plot(True)
save_plot(False)
