import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

eps = 1e-9
     
def f(t, decay=1):
    # return np.exp(-decay*t)
    return decay * ((1 - decay) ** (t-1))

def return_Nf(t, f, theta, data):
    P = data.shape[0]
    M = data.shape[2]
    if t == 0:
        return np.zeros((P,M))
    else:
        return np.sum([f(t-s, theta) * data[:,s,:] for s in range(t)], axis=0)
    
def return_Xf(t, f, theta, X):
    K = X.shape[0]
    if t == 0:
        return np.zeros(K)
    else:
        return np.sum([f(t-s, theta) * X[:,s] for s in range(t)], axis=0)

    
def return_lambda_indpt(alpha, mu, nf, ext):
    # for the first, term, multiply first entry of mu to first row of mu_hat, second entry of
    # mu to second row of mu_hat
    return mu * ext + np.dot(alpha, nf)


def return_T(t, gamma, beta, lambda_indpt, xf, N_mean, N_scale, X_mean, X_scale):
    P = gamma.shape[0]
    M = gamma.shape[1]
    K = gamma.shape[2]
    
    T = np.zeros(shape=(P,M))
    
    for p in range(P):
        for i in range(M):
            T[p,i] = np.sum(gamma[p,i,:] * ((xf - X_mean) / X_scale)) + np.sum(beta[p,:,i,:] * ((np.log(lambda_indpt+1) - N_mean) / N_scale))
    return T

def generate_samples(plist, T_train, T_test, data, X, S, num_samples, P, M, K, base_X=None):
    mu = plist[:P*M].reshape(P,M)
    alpha = plist[P*M:P*M+P*P].reshape(P,P)
    theta = plist[P*M+P*P]
    gamma = plist[P*M+P*P+1:P*M+P*P+P*M*K+1].reshape(P,M,K)
    beta = plist[P*M+P*P+P*M*K+1:P*M+P*P+P*M*K+P*P*M*M+1].reshape(P,P,M,M)

    N_mean = np.mean(np.log(data[0,:,:T_train,:]+1), axis=1)# * 0
    N_scale = (np.std(np.log(data[0,:,:T_train,:]+1), axis=1) + 1e-6)
    if base_X is None:
        X_mean = np.mean(X[:,:T_train], axis=-1)# * 0
        X_scale = (np.std(X[:,:T_train], axis=-1) + 1e-6)
    else:
        X_mean = np.mean(base_X[:,:T_train], axis=-1)
        X_scale = (np.std(base_X[:,:T_train], axis=-1) + 1e-6)     
    new_points = np.zeros(shape=[P, T_test, M, num_samples])
    
    xf_collector = np.zeros(shape=[K, T_test, num_samples])
    lambda_indpt_collector = np.zeros(shape=[P, M, T_test, num_samples])
    s_collector = np.zeros(shape=[P, M, T_test, num_samples])
    
    for ns in range(num_samples):
        new_points[:,0,:,ns] = data[0,:,0,:]

        for t in range(1,T_train):
            actual_data = data[0,:,:t,:]
            rs = []
            
            ext = S[t]
            nf = return_Nf(t, f, theta, actual_data)
            xf = return_Xf(t, f, theta, X)
            lambda_indpt = return_lambda_indpt(alpha, mu, nf, ext)

            
            

            T = return_T(
                t, 
                gamma, 
                beta, 
                lambda_indpt, 
                xf,
                N_mean, N_scale, X_mean, X_scale
                )
            A = np.exp((T.transpose()-T.max(axis=1)).transpose())
            s = (np.divide(A.transpose(), A.sum(axis=1))).transpose()
            lambda_total = lambda_indpt.sum(axis=1)
            lambda_pi = (lambda_total*s.transpose()).transpose()

            xf_collector[:,t,ns] = xf
            lambda_indpt_collector[:,:,t,ns] = lambda_indpt   
            s_collector[:,:,t,ns] = s

            for p in range(P):
                new_point = np.random.poisson(lambda_pi[p,:])
                new_points[p, t, :, ns] = new_point
                rs.append(new_point)
            rs = np.vstack(rs)

        simulated_data = actual_data.copy()
        for t in range(T_train, T_test):
            simulated_data = np.concatenate([simulated_data,
                   rs.reshape(P,1,M)
                  ],axis=1)    

            ext = S[t]
            nf = return_Nf(t, f, theta, simulated_data)
            xf = return_Xf(t, f, theta, X)
            lambda_indpt = return_lambda_indpt(alpha, mu, nf, ext)

            T = return_T(
                t, 
                gamma, 
                beta, 
                lambda_indpt, 
                xf,
                N_mean, N_scale, X_mean, X_scale
                )
            A = np.exp((T.transpose()-T.max(axis=1)).transpose())
            s = (np.divide(A.transpose(), A.sum(axis=1))).transpose()
            lambda_total = lambda_indpt.sum(axis=1)
            lambda_pi = (lambda_total*s.transpose()).transpose()


            xf_collector[:,t,ns] = xf
            lambda_indpt_collector[:,:,t,ns] = lambda_indpt     
            s_collector[:,:,t,ns] = s

            
            rs = []
            for p in range(P): 
                new_point = np.random.poisson(lambda_pi[p,:])
                new_points[p, t, :, ns] = new_point
                rs.append(new_point)
            rs = np.vstack(rs)
            
    return new_points, xf_collector, lambda_indpt_collector, s_collector, X_scale, N_scale
    
def evaluate_fit(data, samples, T_train, T_test):
    train_true = data[0,:,:T_train,:]
    train_pred = np.mean(samples[:, :T_train,:,:], axis=-1)

    test_true = data[0,:,T_train:T_test,:]
    test_pred = np.mean(samples[:, T_train:T_test,:,:], axis=-1)

    level1_train_true = np.sum(train_true,axis=-1)
    level1_train_pred = np.sum(train_pred,axis=-1)

    level1_test_true = np.sum(test_true,axis=-1)
    level1_test_pred = np.sum(test_pred,axis=-1)

    train_level1_smape = np.sum((np.abs(level1_train_true - level1_train_pred) / (level1_train_true + level1_train_pred)) / (T_test-T_train), axis=1)
    test_level1_smape = np.sum((np.abs(level1_test_true - level1_test_pred) / (level1_test_true + level1_test_pred)) / (T_test-T_train), axis=1)    

    train_true_fb = train_true[0,:,:]
    train_true_tw = train_true[1,:,:]
    train_pred_fb = train_pred[0,:,:]
    train_pred_tw = train_pred[1,:,:]

    test_true_fb = test_true[0,:,:]
    test_true_tw = test_true[1,:,:]
    test_pred_fb = test_pred[0,:,:]
    test_pred_tw = test_pred[1,:,:]
    
    train_tw_dist_true = np.divide(train_true_tw.T, train_true_tw.sum(axis=1)).T
    train_tw_dist_pred = np.divide(train_pred_tw.T, train_pred_tw.sum(axis=1)).T
    train_fb_dist_true = np.divide(train_true_fb.T, train_true_fb.sum(axis=1)).T
    train_fb_dist_pred = np.divide(train_pred_fb.T, train_pred_fb.sum(axis=1)).T

    test_tw_dist_true = np.divide(test_true_tw.T, test_true_tw.sum(axis=1)).T
    test_tw_dist_pred = np.divide(test_pred_tw.T, test_pred_tw.sum(axis=1)).T
    test_fb_dist_true = np.divide(test_true_fb.T, test_true_fb.sum(axis=1)).T
    test_fb_dist_pred = np.divide(test_pred_fb.T, test_pred_fb.sum(axis=1)).T
    
    train_tw_entropies_over_t = [entropy(train_tw_dist_true[i,:]+1e-9, train_tw_dist_pred[i,:]+1e-9) for i in range(train_tw_dist_pred.shape[0])]
    train_fb_entropies_over_t = [entropy(train_fb_dist_true[i,:]+1e-9, train_fb_dist_pred[i,:]+1e-9) for i in range(train_fb_dist_true.shape[0])]

    test_tw_entropies_over_t = [entropy(test_tw_dist_true[i,:]+1e-9, test_tw_dist_pred[i,:]+1e-9) for i in range(test_tw_dist_pred.shape[0])]
    test_fb_entropies_over_t = [entropy(test_fb_dist_true[i,:]+1e-9, test_fb_dist_pred[i,:]+1e-9) for i in range(test_fb_dist_true.shape[0])]
    
    train_level2_entropy = 0.5*(np.array(train_tw_entropies_over_t) + np.array(train_fb_entropies_over_t))
    test_level2_entropy = 0.5*(np.array(test_tw_entropies_over_t) + np.array(test_fb_entropies_over_t))

    train_tw_level2_entropy = np.nanmean(train_tw_entropies_over_t)
    train_fb_level2_entropy = np.nanmean(train_fb_entropies_over_t)
    test_tw_level2_entropy = np.nanmean(test_tw_entropies_over_t)
    test_fb_level2_entropy = np.nanmean(test_fb_entropies_over_t)

    return {
        "train_lvl1_fb_smape": train_level1_smape[0],
        "train_lvl1_tw_smape": train_level1_smape[1],
        "train_lvl2_fb_kl": train_fb_level2_entropy,
        "train_lvl2_tw_kl": train_tw_level2_entropy,
        "test_lvl1_fb_smape": test_level1_smape[0],
        "test_lvl1_tw_smape": test_level1_smape[1],
        "test_lvl2_fb_kl": test_fb_level2_entropy,
        "test_lvl2_tw_kl": test_tw_level2_entropy,
        "train_kl": train_level2_entropy,
        "test_kl": test_level2_entropy,
        "train_tw_kl_over_time": train_tw_entropies_over_t,
        "train_fb_kl_over_time": train_fb_entropies_over_t,
        "test_tw_kl_over_time": test_tw_entropies_over_t,
        "test_fb_kl_over_time": test_fb_entropies_over_t
    }


def get_X_elasticities(gamma, std, X, s, P, M, K, T_train):
    el_X = np.zeros(shape=(P, M, K, T_train))
    
    for t in range(T_train):
        for p in range(P):
            for k in range(K):
                for i in range(M):
                    accum = 0
                    for j in range(M):
                        if j == i:
                            accum += (1-s[p,j,t]) * gamma[p,j,k] 
                        else:
                            accum += (-s[p,j,t]) * gamma[p,j,k]
                    el_X[p,i,k,t] = X[k,t] * accum / std[k]
    return el_X

def get_lambda_elasticities(beta, std, L, s, P, M, T_train):
    el_L = np.zeros(shape=(P, P, M, M, T_train))
    
    for t in range(T_train):
        for p in range(P):
            for q in range(P):
                for i in range(M):
                    for j in range(M):
                        accum = 0
                        for k in range(M):
                            if k == i:
                                accum += (1-s[p,k,t]) * beta[p,q,k,j]
                            else:
                                accum += (-s[p,k,t]) * beta[p,q,k,j]
                        el_L[p,q,i,j,t] = L[q,j,t] * accum * (1 / std[q,j]) * (1 / (L[q,j,t] + 1))
    return el_L


def get_nonzero_mean(x, upto):
    means = []
    for k in range(x.shape[0]):
        means.append(np.mean(x[k,:upto]))
    return np.array(means)

def add_val_to_X_one_at_a_time(X, k, val, T_train):
    X_mod = X.copy()
    X_mod[k,T_train:] += val[k]
    return X_mod


def get_average_shares_on_test(samples, T_train):
    fb_estimates = np.mean(samples[0,:,:,:], axis=-1)
    tw_estimates = np.mean(samples[1,:,:,:], axis=-1)
    fb_estimates_df = pd.DataFrame(fb_estimates)
    tw_estimates_df = pd.DataFrame(tw_estimates)
    fbedf_normalized = fb_estimates_df.div(fb_estimates_df.sum(axis=1),axis=0).values
    twitteredf_normalized = tw_estimates_df.div(tw_estimates_df.sum(axis=1),axis=0).values
    
    return np.mean(fbedf_normalized[T_train:,:], axis=0), np.mean(twitteredf_normalized[T_train:,:], axis=0)
