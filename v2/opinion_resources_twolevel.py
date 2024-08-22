import pickle
import numpy as np
import pandas as pd

from opinion_resources_opt1 import (
    run_optimization_many_sequences_given_starting_point as run_opt1
)
from opinion_resources_opt2 import (
    run_optimization_many_sequences_given_starting_point as run_opt2
)

from opinion_analysis_v2 import (
    generate_samples,
    evaluate_fit,
    get_X_elasticities,
    get_lambda_elasticities,
    add_val_to_X_one_at_a_time,
    get_average_shares_on_test,
    get_nonzero_mean
)

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
    X_volume
):
    res_opt1 = run_opt1(
        p0_opt1,
        S,
        samples,
        0, #regularization_parameters[0],
        time_averaged,
        P,
        how_long,
        how_many_samples,
        logfit_label
    )
    
    mu = res_opt1[0][:P]
    alpha = res_opt1[0][P:P+P*P].reshape(P,P)
    theta = res_opt1[0][-1]
    
    res_opt2 = run_opt2(
        p0_opt2,
        samples,
        X,
        S,
        alpha,
        mu,
        theta,
        regularization_parameters[1],
        regularization_parameters[0], # replace level 1 reg with leg 2 structural reg
        time_averaged,
        P,
        M,
        K,
        how_long,
        how_many_samples,
        logfit_label  
    )

    mu = res_opt1[0][:P]
    alpha = res_opt1[0][P:P+P*P]
    theta = res_opt1[0][-1]

    mu_hat = res_opt2[0][:P*M].reshape(P,M)
    mu_opinion_level = (mu_hat.T*mu).T.reshape(-1)

    xopt = np.hstack([
        mu_opinion_level, 
        alpha,
        theta,
        res_opt2[0][P*M:]
    ])
    
    xopt_orig = xopt.copy()
    
    try:
        # get predictions
        predictions, xf, lindpt, s, X_scale, N_scale = generate_samples(xopt, how_long, how_long_test, samples, X, S, how_many_prediction_samples, P, M, K)
        
        # fit metrics
        fit_metrics = evaluate_fit(samples, predictions, how_long, how_long_test)

    except:
        predictions = np.nan
        fit_metrics = np.nan
    
    gamma = xopt[-P*P*M*M-P*M*K:-P*P*M*M].reshape(P,M,K).copy()
    beta = xopt[-P*P*M*M:].reshape(P,P,M,M).copy()
    
    try:
        # get elasticities
        X_elasticities = get_X_elasticities(
            gamma[:,[0,2,4,6,8,10,1,3,5,7,9,11],:],
            np.std(X, axis=1),
            np.mean(xf, axis=-1),
            np.mean(s, axis=-1)[:,[0,2,4,6,8,10,1,3,5,7,9,11],:],
            P, M, K, how_long
        )

        rr=pd.DataFrame(beta[0,0])
        beta[0,0] = rr.loc[[0,2,4,6,8,10,1,3,5,7,9,11],[0,2,4,6,8,10,1,3,5,7,9,11]]
        rr=pd.DataFrame(beta[0,1])
        beta[0,1] = rr.loc[[0,2,4,6,8,10,1,3,5,7,9,11],[0,2,4,6,8,10,1,3,5,7,9,11]]
        rr=pd.DataFrame(beta[1,0])
        beta[1,0] = rr.loc[[0,2,4,6,8,10,1,3,5,7,9,11],[0,2,4,6,8,10,1,3,5,7,9,11]]
        rr=pd.DataFrame(beta[1,1])
        beta[1,1] = rr.loc[[0,2,4,6,8,10,1,3,5,7,9,11],[0,2,4,6,8,10,1,3,5,7,9,11]]
        L_elasticities = get_lambda_elasticities(
            beta,
            N_scale[:,[0,2,4,6,8,10,1,3,5,7,9,11]],
            np.mean(lindpt, axis=-1)[:,[0,2,4,6,8,10,1,3,5,7,9,11],:],
            np.mean(s, axis=-1)[:,[0,2,4,6,8,10,1,3,5,7,9,11],:],
            P, M, how_long
        )
    except:
        X_elasticities = np.nan
        L_elasticities = np.nan

    to_vary2 = [-1., -.5, -.25, -.1, -0.05, 0, .05, .1, .25, .5, 1.]
    
#     # mean on train period
    mean_vals = get_nonzero_mean(X_volume, how_long)
    
    n_list = [mean_vals*x for x in to_vary2]
    num_samples_for_perc_inc = 50

    av_dist_on_test_over_K = []
    for k in range(K):
        try:
            av_dist_on_test = []
            for n in n_list:
                Xx = add_val_to_X_one_at_a_time(X, k, n, how_long)
                predictions_under_Xx, _, _, _, _, _ = generate_samples(xopt, how_long, how_long_test, samples, Xx, S, num_samples_for_perc_inc, P, M, K, base_X = X)
                av_dist_on_test.append(get_average_shares_on_test(predictions_under_Xx, how_long))
        except:
            av_dist_on_test = np.nan
        av_dist_on_test_over_K.append(av_dist_on_test)
        
    return [xopt, res_opt1, res_opt2, predictions, fit_metrics, X_elasticities, L_elasticities, av_dist_on_test_over_K, np.all(xopt_orig==xopt)]


def perform_two_level_analysis_2(
    idx,
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
    X_volume
):
    res_opt1 = "done"
    
    mu = res_opt1[0][:P]
    alpha = res_opt1[0][P:P+P*P].reshape(P,P)
    theta = res_opt1[0][-1]
    
    res_opt2 = "done"

    xoptdat = pickle.load(open("hypertuning/2D_T0_h.p", "rb"))

    xopt = xoptdat[idx][1]
    xopt_orig = xopt.copy()
    
    try:
        # get predictions
        predictions, xf, lindpt, s, X_scale, N_scale = generate_samples(xopt, how_long, how_long_test, samples, X, S, how_many_prediction_samples, P, M, K)
        
        # fit metrics
        fit_metrics = evaluate_fit(samples, predictions, how_long, how_long_test)

    except:
        predictions = np.nan
        fit_metrics = np.nan
    
    gamma = xopt[-P*P*M*M-P*M*K:-P*P*M*M].reshape(P,M,K).copy()
    beta = xopt[-P*P*M*M:].reshape(P,P,M,M).copy()
    
    try:
        # get elasticities
        X_elasticities = get_X_elasticities(
            gamma[:,[0,2,4,6,8,10,1,3,5,7,9,11],:],
            np.std(X, axis=1),
            np.mean(xf, axis=-1),
            np.mean(s, axis=-1)[:,[0,2,4,6,8,10,1,3,5,7,9,11],:],
            P, M, K, how_long
        )

        rr=pd.DataFrame(beta[0,0])
        beta[0,0] = rr.loc[[0,2,4,6,8,10,1,3,5,7,9,11],[0,2,4,6,8,10,1,3,5,7,9,11]]
        rr=pd.DataFrame(beta[0,1])
        beta[0,1] = rr.loc[[0,2,4,6,8,10,1,3,5,7,9,11],[0,2,4,6,8,10,1,3,5,7,9,11]]
        rr=pd.DataFrame(beta[1,0])
        beta[1,0] = rr.loc[[0,2,4,6,8,10,1,3,5,7,9,11],[0,2,4,6,8,10,1,3,5,7,9,11]]
        rr=pd.DataFrame(beta[1,1])
        beta[1,1] = rr.loc[[0,2,4,6,8,10,1,3,5,7,9,11],[0,2,4,6,8,10,1,3,5,7,9,11]]
        L_elasticities = get_lambda_elasticities(
            beta,
            N_scale[:,[0,2,4,6,8,10,1,3,5,7,9,11]],
            np.mean(lindpt, axis=-1)[:,[0,2,4,6,8,10,1,3,5,7,9,11],:],
            np.mean(s, axis=-1)[:,[0,2,4,6,8,10,1,3,5,7,9,11],:],
            P, M, how_long
        )
    except:
        X_elasticities = np.nan
        L_elasticities = np.nan

    to_vary2 = [-1., -.5, -.25, -.1, -0.05, 0, 0.05, .1, .25, .5, 1.]
    # to_vary2 = [0]
    
    mean_vals = get_nonzero_mean(X_volume, how_long)
    
    n_list = [mean_vals*x for x in to_vary2]
    num_samples_for_perc_inc = 50

    av_dist_on_test_over_K = []
    for k in range(K):
        try:
            av_dist_on_test = []
            for n in n_list:
                Xx = add_val_to_X_one_at_a_time(X, k, n, how_long)
                predictions_under_Xx, _, _, _, _, _ = generate_samples(xopt, how_long, how_long_test, samples, Xx, S, num_samples_for_perc_inc, P, M, K, base_X = X)
                av_dist_on_test.append(get_average_shares_on_test(predictions_under_Xx, how_long))
        except:
            av_dist_on_test = np.nan
        av_dist_on_test_over_K.append(av_dist_on_test)
        
    return [xopt, res_opt1, res_opt2, predictions, fit_metrics, X_elasticities, L_elasticities, av_dist_on_test_over_K, np.all(xopt_orig==xopt)]

