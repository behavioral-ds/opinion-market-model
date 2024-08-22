import pickle
import sys, os, copy
import pandas as pd
import numpy as np
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter, sleep
from itertools import product

from opinion_resources_opt1 import (
    return_random_starting_points as return_starting_points_opt1,
)
from opinion_resources_opt2 import (
    return_random_starting_points as return_starting_points_opt2,
)
from opinion_resources_twolevel import perform_two_level_analysis

t_start = perf_counter()

how_many_samples = 1
num_starting_points = 3
max_workers = 48
eps = 1e-9
how_many_prediction_samples = 5

P = 2  # number of platforms
M = 12  # number of opinions on each platform
K = 12  # number of interventions

# load sample dataset
samples = pickle.load(
    open("samples/us_and_au_reducedopinionset_hourly_farright.p", "rb")
)

# load exogenous signal
S = pickle.load(open("samples/googletrends.p", "rb"))
S = np.concatenate([[x] * 24 for x in S])

# load interventions
X = pickle.load(open("samples/RELIABLE_UNRELIABLE_K12.p", "rb"))
X = X[list(range(K)), :].astype(float)

regularization_list = list(
    product(
        [0],  # level 1 regularization parameter
        [0.1],  # level 2 regularization parameter
    )
)

# training interval, 75 days x 24 hours
T_list = [75 * 24]
# test interval, 90 days x 24 hours
T_test_list = [90 * 24]


def main():

    if len(sys.argv) != 2:
        sys.exit()
    try:
        T_index = int(sys.argv[1])
    except:
        print("One of your arguments was not an integer.")
        sys.exit()

    # set train and test intervals
    how_long = T_list[T_index]
    how_long_test = T_test_list[T_index]

    opt1_starting_points = return_starting_points_opt1(num_starting_points, samples, P)
    opt2_starting_points = return_starting_points_opt2(num_starting_points, P, M, K)
    hyparam_x0index_list = list(
        product(regularization_list, range(num_starting_points))
    )

    sim_output_collector = []
    fit_duration_collector = []

    r_regularization = [x[0] for x in hyparam_x0index_list]
    r_opt_1_x0 = [opt1_starting_points[x[1]] for x in hyparam_x0index_list]
    r_opt_2_x0 = [opt2_starting_points[x[1]] for x in hyparam_x0index_list]
    r_timeaveraged = [True] * len(hyparam_x0index_list)
    (
        r_samples,
        r_X,
        r_P,
        r_M,
        r_K,
        r_howlong,
        r_howlongtest,
        r_howmanysamples,
        r_howmanypredictionsamples,
        r_S,
    ) = list(
        zip(
            *repeat(
                [
                    samples,
                    X,
                    P,
                    M,
                    K,
                    how_long,
                    how_long_test,
                    how_many_samples,
                    how_many_prediction_samples,
                    S,
                ],
                len(hyparam_x0index_list),
            )
        )
    )
    r_logfitlabel = [
        "_".join(["T" + str(how_long), "reg" + str(x[0]), "start" + str(x[1])])
        for x in hyparam_x0index_list
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        sim_outputs = executor.map(
            perform_two_level_analysis,
            r_opt_1_x0,
            r_opt_2_x0,
            r_samples,
            r_X,
            r_S,
            r_regularization,
            r_timeaveraged,
            r_P,
            r_M,
            r_K,
            r_howlong,
            r_howlongtest,
            r_howmanysamples,
            r_howmanypredictionsamples,
            r_logfitlabel,
        )

    sim_outputs = list(sim_outputs)
    fit_duration = perf_counter() - t_start

    sim_output_collector.append(sim_outputs)
    fit_duration_collector.append(fit_duration)

    pickle.dump(
        [sim_output_collector, fit_duration_collector],
        open(f"hypertuning/T{T_index}_h.p", "wb"),
    )


if __name__ == "__main__":
    main()
