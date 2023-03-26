import numpy as np
import pandas as pd

from opinion_resources_opt1 import (
    run_optimization_many_sequences_given_starting_point as run_opt1,
)
from opinion_resources_opt2 import (
    run_optimization_many_sequences_given_starting_point as run_opt2,
)

from opinion_analysis_v2 import (
    generate_samples,
    evaluate_fit,
    get_X_elasticities,
    get_lambda_elasticities,
    add_val_to_X_one_at_a_time,
    get_average_shares_on_test,
)

# IMPLEMENTATION ASSUMES PLATFORM-DEPENDENT STRUCTURE ON GAMMA ESTIMATES VIA REGULARIZATION SPECIFIC TO THE BUSHFIRE CASE STUDY (SEE SUPPLEMENTARY MATERIAL).

# FOR GENERAL APPLICATIONS, MODIFY OR REMOVE THE REGULARIZATION TERMS.


def perform_two_level_analysis(
    p0_opt1,
    p0_opt2,
    samples,
    X,
    S,
    regularization_parameters,
    time_averaged,
    P,
    M,
    K,
    how_long,
    how_long_test,
    how_many_samples,
    how_many_prediction_samples,
    logfit_label,
):
    # fit level 1 model
    res_opt1 = run_opt1(
        p0_opt1,
        S,
        samples,
        regularization_parameters[0],
        time_averaged,
        P,
        how_long,
        how_many_samples,
        logfit_label,
    )

    mu = res_opt1[0][:P]
    alpha = res_opt1[0][P : P + P * P].reshape(P, P)
    theta = res_opt1[0][-1]

    # fit level 2 model
    res_opt2 = run_opt2(
        p0_opt2,
        samples,
        X,
        S,
        alpha,
        mu,
        theta,
        regularization_parameters[1],
        time_averaged,
        P,
        M,
        K,
        how_long,
        how_many_samples,
        logfit_label,
    )

    mu = res_opt1[0][:P]
    alpha = res_opt1[0][P : P + P * P]
    theta = res_opt1[0][-1]

    mu_hat = res_opt2[0][: P * M].reshape(P, M)
    mu_opinion_level = (mu_hat.T * mu).T.reshape(-1)

    xopt = np.hstack([mu_opinion_level, alpha, theta, res_opt2[0][P * M :]])

    xopt_orig = xopt.copy()

    try:
        # get predictions
        predictions, xf, lindpt, s, X_scale, N_scale = generate_samples(
            xopt,
            how_long,
            how_long_test,
            samples,
            X,
            S,
            how_many_prediction_samples,
            P,
            M,
            K,
        )

        # fit metrics
        fit_metrics = evaluate_fit(samples, predictions, how_long, how_long_test)

    except:
        predictions = np.nan
        fit_metrics = np.nan

    gamma = xopt[-P * P * M * M - P * M * K : -P * P * M * M].reshape(P, M, K).copy()
    beta = xopt[-P * P * M * M :].reshape(P, P, M, M).copy()

    try:
        # calculate X elasticities
        X_elasticities = get_X_elasticities(
            gamma[:, [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11], :],
            np.std(X, axis=1),
            np.mean(xf, axis=-1),
            np.mean(s, axis=-1)[:, [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11], :],
            P,
            M,
            K,
            how_long,
        )  # we re-order the axes so that C and R columns are all clumped together.

        rr = pd.DataFrame(beta[0, 0])
        beta[0, 0] = rr.loc[
            [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11],
            [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11],
        ]
        rr = pd.DataFrame(beta[0, 1])
        beta[0, 1] = rr.loc[
            [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11],
            [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11],
        ]
        rr = pd.DataFrame(beta[1, 0])
        beta[1, 0] = rr.loc[
            [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11],
            [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11],
        ]
        rr = pd.DataFrame(beta[1, 1])
        beta[1, 1] = rr.loc[
            [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11],
            [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11],
        ]

        # calculate L elasticities
        L_elasticities = get_lambda_elasticities(
            beta,
            N_scale[:, [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]],
            np.mean(lindpt, axis=-1)[:, [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11], :],
            np.mean(s, axis=-1)[:, [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11], :],
            P,
            M,
            how_long,
        )
    except:
        X_elasticities = np.nan
        L_elasticities = np.nan

    # GET AVERAGE OPINION SHARES ON TEST (FOLLOWING THE WHAT IF ANALYSIS IN MAIN TEXT)
    to_vary2 = [-0.5, -0.25, -0.1, -0.05, 0, 0.05, 0.1, 0.25, 0.5]
    # mean on train period
    mean_vals = X[:, :how_long].mean(axis=1)
    n_list = [mean_vals * x for x in to_vary2]
    num_samples_for_perc_inc = 5

    av_dist_on_test_over_K = []
    for k in range(K):
        try:
            av_dist_on_test = []
            for n in n_list:
                Xx = add_val_to_X_one_at_a_time(X, k, n, how_long)
                predictions_under_Xx, _, _, _, _, _ = generate_samples(
                    xopt,
                    how_long,
                    how_long_test,
                    samples,
                    Xx,
                    S,
                    num_samples_for_perc_inc,
                    P,
                    M,
                    K,
                    base_X=X,
                )
                av_dist_on_test.append(
                    get_average_shares_on_test(predictions_under_Xx, how_long)
                )
        except:
            av_dist_on_test = np.nan
        av_dist_on_test_over_K.append(av_dist_on_test)

    return [
        xopt,
        res_opt1,
        res_opt2,
        predictions,
        fit_metrics,
        X_elasticities,
        L_elasticities,
        av_dist_on_test_over_K,
        np.all(xopt_orig == xopt),
    ]
