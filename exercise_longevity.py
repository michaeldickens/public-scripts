"""

exercise_longevity.py
---------------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2024-08-30

Estimate how much exercise improves life expectancy.

Data sources
------------

- exercise-RR-cancer.tsv: Morishita, S., Hamaue, Y., Fukushima, T., Tanaka, T., Fu, J. B., & Nakano, J. (2020). Effect of Exercise on Mortality and Recurrence in Patients With Cancer: A Systematic Review and Meta-Analysis. https://doi.org/10.1177/1534735420917462
- exercise-RR-post-MI.tsv: Lawler, P. R., Filion, K. B., & Eisenberg, M. J. (2011). Efficacy of exercise-based cardiac rehabilitation post–myocardial infarction: A systematic review and meta-analysis of randomized controlled trials. http://doi.org/10.1016/j.ahj.2011.07.017
- MI-life-expectancy.tsv: Bucholz, E. M., Normand, S. L. T., Wang, Y., Ma, S., Lin, H., & Krumholz, H. M. (2015). Life Expectancy and Years of Potential Life Lost After Acute Myocardial Infarction by Sex and Race. https://doi.org/10.1016/j.jacc.2015.06.022

"""

from pprint import pprint
from matplotlib import pyplot as plt


LIFE_EXPECTANCY_GIVEN_DIE_THIS_YEAR = 0.5


def read_mi_life_expectancy():
    # based on 2010 US Census data because that's approximately when the data
    # was collected
    pop_demo_proportions = {
        "White": 0.64,
        "Black": 0.12,
        "Men": 0.495,
        "Women": 0.505,
    }
    life_expectancy_by_demo = {}

    with open("data/MI-life-expectancy.tsv", "r") as f:
        header = None
        for line in f:
            fields = line.strip().split("\t")
            state = fields[0]
            if header is None:
                header = fields
                assert header[0] == "Age (years)"
                for i in range(1, len(header)):
                    life_expectancy_by_demo[header[i]] = {}
                continue
            age = int(fields[0])
            for i in range(1, len(header)):
                life_expectancy_by_demo[header[i]][age] = float(fields[i])

    life_expectancy_by_demo["Overall"] = {}
    for age in life_expectancy_by_demo["Black Men"]:
        life_expectancy_by_demo["Overall"][age] = (
            life_expectancy_by_demo["Black Men"][age] *
            pop_demo_proportions["Black"] * pop_demo_proportions["Men"]
            + life_expectancy_by_demo["Black Women"][age]
            * pop_demo_proportions["Black"] * pop_demo_proportions["Women"]
            + life_expectancy_by_demo["White Men"][age]
            * pop_demo_proportions["White"] * pop_demo_proportions["Men"]
            + life_expectancy_by_demo["White Women"][age]
            * pop_demo_proportions["White"] * pop_demo_proportions["Women"]
        ) / (
            pop_demo_proportions["Black"] + pop_demo_proportions["White"]
        )

    return life_expectancy_by_demo


def read_ssa_projections():
    projected_mortality_tables = {}
    for gender in ["male", "female"]:
        gender_table = {}
        with open(f"data/SSA projected mortality {gender}_TR2024.csv", "r") as f:
            lines = f.readlines()
            intro_line = lines[0]
            header = lines[1].strip().split(",")
            for line in lines[2:]:
                fields = line.strip().split(",")
                year = int(fields[0])
                gender_table[year] = {}
                for i in range(1, len(fields)):
                    gender_table[year][int(header[i])] = float(fields[i])

        projected_mortality_tables[gender] = gender_table
    return projected_mortality_tables


def fetch_annual_mortality(projected_mortality_tables, birth_year):
    assert birth_year >= 2022 - 119, f"Birth year {birth_year} is too old for this data"
    assert birth_year <= 2099, f"Birth year {birth_year} is too young for this data"

    p_male = 0.5
    mortality_table = {}
    for year in range(max(birth_year, 2024), min(birth_year + 119, 2099) + 1):
        age = year - birth_year
        p_death_male = projected_mortality_tables["male"][year][age]
        p_death_female = projected_mortality_tables["female"][year][age]
        mortality_table[age] = p_death_male * p_male + p_death_female * (1 - p_male)

        odds_ratio_male = (1 - p_death_male) / (1 - p_death_female)
        p_male = odds_ratio_male / (1 + odds_ratio_male)

    return mortality_table


def calculate_annual_mortality(life_expectancy):
    annual_mortality = {}
    for age in list(life_expectancy.keys())[:-1]:
        curr_life_expectancy = life_expectancy[age]
        next_life_expectancy = 1 + life_expectancy[age + 1]

        p_dying_this_year = (
            (curr_life_expectancy - next_life_expectancy)
            / (LIFE_EXPECTANCY_GIVEN_DIE_THIS_YEAR - next_life_expectancy)
        )
        annual_mortality[age] = p_dying_this_year
    return annual_mortality


def calculate_life_expectancy(annual_mortality, max_age_LE):
    reverse_ages = list(reversed(annual_mortality.keys()))
    life_expectancy = {}
    life_expectancy[reverse_ages[0] + 1] = max_age_LE
    for age in reverse_ages:
        p_dying_this_year = annual_mortality[age]
        life_expectancy[age] = (
            p_dying_this_year * LIFE_EXPECTANCY_GIVEN_DIE_THIS_YEAR
            + (1 - p_dying_this_year) * (1 + life_expectancy[age + 1])
        )
    del life_expectancy[reverse_ages[0] + 1]
    return {
        age: life_expectancy[age] for age in annual_mortality.keys()
    }


def read_exercise_rr_helper(filename):
    exercise_studies = []

    with open(filename, "r") as f:
        header = None
        for line in f:
            fields = line.strip().split("\t")
            if header is None:
                header = fields
                continue
            study = {header[0]: fields[0]}
            for i in range(1, len(header)):
                study[header[i]] = float(fields[i])

            exercise_studies.append(study)

    return exercise_studies


def read_exercise_rr_post_mi():
    return read_exercise_rr_helper("data/exercise-RR-post-MI.tsv")


def read_exercise_rr_cancer():
    return read_exercise_rr_helper("data/exercise-RR-cancer.tsv")


def life_expectancy_with_exercise(life_expectancy, RR):
    mortality = calculate_annual_mortality(life_expectancy)
    exercise_mortality = {
        age: mortality[age] * RR for age in mortality
    }
    exercise_LE = calculate_life_expectancy(exercise_mortality, life_expectancy[max(life_expectancy.keys())])
    return {
        age: exercise_LE[age] for age in life_expectancy if age in exercise_LE
    }



def mi_stuff():
    exercise_mi = read_exercise_rr_post_mi()
    duration = 0
    RR = 0
    total_weight = 0
    for study in exercise_mi:
        weight = study["Weight"]
        study_duration = study["Follow-up duration (m)"]
        total_weight += weight
        duration += study_duration * weight
        RR += study["RR"] * weight

    RR /= total_weight
    avg_duration = duration / total_weight
    RR_annualized = RR ** (12 / avg_duration)
    print(f"Reported overall RR: 0.74")
    print(f"Weighted RR: {RR:.2f}")
    print(f"Average duration: {avg_duration:.2f} months")
    print(f"Weighted RR (outer-annualized): {RR_annualized:.3f}")
    print(f"Weighted RR (cheaty outer-annualized): {0.74 ** (12 / avg_duration):.3f}")

def cancer_stuff():
    exercise_cancer = read_exercise_rr_cancer()
    intervention_mortality = 0
    control_mortality = 0
    intervention_total = 0
    control_total = 0
    intervention_mortality_annualized = 0
    control_mortality_annualized = 0
    duration = 0
    for study in exercise_cancer:
        weight = study["Weight"]
        study_duration = study["Observation Period (m)"]
        intervention_mortality += study["Experimental Mortality"] * weight
        control_mortality += study["Control Mortality"] * weight
        intervention_total += study["Experimental Total"] * weight
        control_total += study["Control Total"] * weight
        duration += study_duration * weight
        intervention_mortality_annualized += study["Experimental Mortality"] * weight * 12 / study_duration
        control_mortality_annualized += study["Control Mortality"] * weight * 12 / study_duration

    avg_duration = duration / sum(study["Weight"] for study in exercise_cancer)
    intervention_mortality_rate = intervention_mortality / intervention_total
    control_mortality_rate = control_mortality / control_total

    # study says 0.76, my calculation says 0.78... eh close enough
    RR = intervention_mortality_rate / control_mortality_rate
    RR_annualized_inner = (intervention_mortality_annualized / intervention_total) / (control_mortality_annualized / control_total)
    RR_annualized_outer = (intervention_mortality_rate ** (12 / avg_duration)) / (control_mortality_rate ** (12 / avg_duration))
    RR_annualized_outer = RR ** (12 / avg_duration)
    print(f"Intervention mortality rate: {intervention_mortality_rate:.2f}")
    print(f"Control mortality rate: {control_mortality_rate:.2f}")
    print(f"Average duration: {avg_duration:.2f} months")
    print(f"Weighted RR: {RR:.2f}")

    # TODO: does it actually make sense to annualize the RR? I think so.
    # basically if in (say) 24 months, 100 people die, then in 12 months, 50
    # people die. the number you actually want is 50 / n, aggregated across
    # every study ("inner annualized").
    #
    # I'm not sure RR^(1/duration) ("outer annualized") is a good
    # approximation, it might be a coincidence. the formula for a 2-study combo
    # is
    #
    # (ei1 * d1 + ei2 * d2) * (nc1 + nc2)
    # -----------------------------------
    # (ni1 + ni2) * (ec1 * d1 + ec2 * d2)
    #
    # (e = events, n = total, c = control, i = intervention, d = duration)
    #
    # but inner-annualized RR might actually be worse than regular RR because
    # it changes the weights. maybe we just want regular RR. for a single
    # study, duration doesn't change RR. but I think inner-annualized RR is the
    # right way to do it because you want to weight the numerator and
    # denominator separately, not weight the RR
    print(f"Weighted RR (inner-annualized): {RR_annualized_inner:.3f}")
    print(f"Weighted RR (outer-annualized): {RR_annualized_outer:.3f}")


if __name__ == "__main__":
    # mi_stuff()
    ssa_projections = read_ssa_projections()
    annual_mortality = fetch_annual_mortality(ssa_projections, 1994)
    life_expectancy = calculate_life_expectancy(annual_mortality, 0)
    # exercise_LE = life_expectancy_with_exercise(life_expectancy, 0.74)  # RR for cancer

    LE_for_RR = {}
    LE_improvement_for_RR = {}
    for RR_percent in range(60, 101):
        RR = RR_percent / 100
        exercise_LE = life_expectancy_with_exercise(life_expectancy, RR)
        LE_for_RR[RR] = exercise_LE
        LE_improvement_for_RR[RR] = {}
        for age in range(30, 125):
            if age in exercise_LE:
                LE_improvement = exercise_LE[age] - life_expectancy[age]
                LE_improvement_for_RR[RR][age] = LE_improvement

    # set axes
    plt.figure(1)
    plt.xlabel("RR")
    plt.ylabel("Life expectancy improvement (years)")
    plt.plot(LE_improvement_for_RR.keys(), [LE_improvement_for_RR[RR][30] for RR in LE_improvement_for_RR], label="30")
    plt.plot(LE_improvement_for_RR.keys(), [LE_improvement_for_RR[RR][40] for RR in LE_improvement_for_RR], label="40")
    plt.plot(LE_improvement_for_RR.keys(), [LE_improvement_for_RR[RR][50] for RR in LE_improvement_for_RR], label="50")
    plt.plot(LE_improvement_for_RR.keys(), [LE_improvement_for_RR[RR][60] for RR in LE_improvement_for_RR], label="60")
    plt.plot(LE_improvement_for_RR.keys(), [LE_improvement_for_RR[RR][70] for RR in LE_improvement_for_RR], label="70")
    plt.plot(LE_improvement_for_RR.keys(), [LE_improvement_for_RR[RR][80] for RR in LE_improvement_for_RR], label="80")
    plt.legend()

    plt.figure(2)
    plt.xlabel("RR")
    plt.ylabel("Life expectancy improvement per year of exercise")
    plt.plot(LE_improvement_for_RR.keys(), [LE_improvement_for_RR[RR][30] / LE_for_RR[RR][30] for RR in LE_improvement_for_RR], label="30")
    plt.plot(LE_improvement_for_RR.keys(), [LE_improvement_for_RR[RR][40] / LE_for_RR[RR][40] for RR in LE_improvement_for_RR], label="40")
    plt.plot(LE_improvement_for_RR.keys(), [LE_improvement_for_RR[RR][50] / LE_for_RR[RR][50] for RR in LE_improvement_for_RR], label="50")
    plt.plot(LE_improvement_for_RR.keys(), [LE_improvement_for_RR[RR][60] / LE_for_RR[RR][60] for RR in LE_improvement_for_RR], label="60")
    plt.plot(LE_improvement_for_RR.keys(), [LE_improvement_for_RR[RR][70] / LE_for_RR[RR][70] for RR in LE_improvement_for_RR], label="70")
    plt.plot(LE_improvement_for_RR.keys(), [LE_improvement_for_RR[RR][80] / LE_for_RR[RR][80] for RR in LE_improvement_for_RR], label="80")
    plt.legend()

    plt.show()
