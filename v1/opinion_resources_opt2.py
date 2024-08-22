import ipopt, logging
import numpy as np
import pickle
from time import perf_counter, sleep
from numpy.random import RandomState

# IMPLEMENTATION ASSUMES PLATFORM-DEPENDENT STRUCTURE ON GAMMA ESTIMATES VIA REGULARIZATION SPECIFIC TO THE BUSHFIRE CASE STUDY (SEE SUPPLEMENTARY MATERIAL).

# FOR GENERAL APPLICATIONS, MODIFY OR REMOVE THE REGULARIZATION TERMS.

eps = 1e-9


def filter_on_i(i, K):
    # filtering matrix. if i=0,1 -> k=0; i=2,3 -> k=1, etc.
    filt = np.zeros(K)
    filt[i // 2] = 1
    filt[int((i // 2) + (K / 2))] = 1
    return filt


def flatten_parameters(mu_hat, gamma, beta):
    return np.hstack([mu_hat.reshape(-1), gamma.reshape(-1), beta.reshape(-1)])


def unflatten_parameters(plist, P, M, K):
    mu_hat = plist[: P * M].reshape(P, M)
    gamma = plist[P * M : P * M + P * M * K].reshape(P, M, K)
    beta = plist[P * M + P * M * K : P * M + P * M * K + P * P * M * M].reshape(
        P, P, M, M
    )
    return mu_hat, gamma, beta


def f(t, theta):
    return theta * ((1 - theta) ** (t - 1))


def return_Nf(t, f, theta, data):
    P = data.shape[0]
    M = data.shape[2]
    if t == 0:
        return np.zeros((P, M))
    else:
        return np.sum([f(t - s, theta) * data[:, s, :] for s in range(t)], axis=0)


def return_Xf(t, f, theta, X):
    K = X.shape[0]
    if t == 0:
        return np.zeros(K)
    else:
        return np.sum([f(t - s, theta) * X[:, s] for s in range(t)], axis=0)


def return_lambda_indpt(alpha, mu, mu_hat, nf, ext):
    # for the first, term, multiply first entry of mu to first row of mu_hat, second entry of
    # mu to second row of mu_hat
    return ((mu_hat.T * mu).T) * ext + np.dot(alpha, nf)


def return_T(t, gamma, beta, lambda_indpt, xf, N_mean, N_scale, X_mean, X_scale):
    P = gamma.shape[0]
    M = gamma.shape[1]
    K = gamma.shape[2]

    T = np.zeros(shape=(P, M))
    for p in range(P):
        for i in range(M):
            T[p, i] = np.sum(
                gamma[p, i, :] * ((xf - X_mean) / X_scale) * filter_on_i(i, K)
            ) + np.sum(
                beta[p, :, i, :] * ((np.log(lambda_indpt + 1) - N_mean) / N_scale)
            )
    return T


def return_gradient_of_lambda_p(p, t, mu_hat, gamma, beta, nf, ext, data, alpha, mu):

    # order is mu_hat, gamma, beta

    P = data.shape[0]
    T = data.shape[1]
    M = data.shape[2]
    K = gamma.shape[2]

    len_mu_hat = P * M
    len_gamma = P * M * K
    len_beta = P * P * M * M

    gradient = np.zeros(shape=len_mu_hat + len_gamma + len_beta)
    gradient[p * M : p * M + M] = mu[p] * ext
    return gradient


def return_gradient_of_T_p_i_wrt_mu_hat(
    i, p, mu_hat, beta, lambda_indpt, mu, ext, N_mean, N_scale
):
    collector = np.zeros(shape=mu_hat.shape)

    P = beta.shape[0]
    M = beta.shape[-1]

    for q in range(P):
        for j in range(M):
            collector[q, j] = (
                (beta[p, q, i, j] / N_scale[q, j])
                * (1 / (lambda_indpt[q, j] + 1))
                * mu[q]
                * ext
            )
    return collector.reshape(-1)


def return_gradient_of_T_p_i_wrt_gamma(i, p, t, gamma, xf, X_mean, X_scale):
    collector = np.zeros(shape=gamma.shape)

    P = gamma.shape[0]
    M = gamma.shape[1]
    K = gamma.shape[2]

    for k in range(K):
        if (k == i // 2) or (k == int((i // 2) + (K / 2))):  # nonzero only if k==i//2
            collector[p, i, k] = (xf[k] - X_mean[k]) / X_scale[k]
    return collector.reshape(-1)


def return_gradient_of_T_p_wrt_beta(
    i, p, t, mu_hat, beta, lambda_indpt, N_mean, N_scale
):
    collector = np.zeros(shape=beta.shape)

    P = beta.shape[0]
    M = beta.shape[-1]

    for r in range(P):
        for k in range(M):
            collector[p, r, i, k] = (
                np.log(lambda_indpt[r, k] + 1) - N_mean[r, k]
            ) / N_scale[r, k]
    return collector.reshape(-1)


def return_gradient_of_T_p_i(
    i,
    p,
    t,
    mu_hat,
    beta,
    gamma,
    xf,
    lambda_indpt,
    mu,
    ext,
    N_mean,
    N_scale,
    X_mean,
    X_scale,
):
    wrt_mu_hat = return_gradient_of_T_p_i_wrt_mu_hat(
        i, p, mu_hat, beta, lambda_indpt, mu, ext, N_mean, N_scale
    )
    wrt_gamma = return_gradient_of_T_p_i_wrt_gamma(i, p, t, gamma, xf, X_mean, X_scale)
    wrt_beta = return_gradient_of_T_p_wrt_beta(
        i, p, t, mu_hat, beta, lambda_indpt, N_mean, N_scale
    )
    return np.hstack([wrt_mu_hat, wrt_gamma, wrt_beta])


def return_likelihood_and_gradient(
    mu_hat,
    gamma,
    beta,
    data,
    X,
    S,
    alpha,
    mu,
    theta,
    regularization,
    time_averaged=True,
):

    P = gamma.shape[0]
    M = gamma.shape[1]

    ll_scaling = np.array([1, 1])
    N_mean = np.mean(np.log(data + 1), axis=1)
    N_scale = np.std(np.log(data + 1), axis=1) + 1e-6
    X_mean = np.mean(X, axis=-1)
    X_scale = np.std(X, axis=-1) + 1e-6

    log_likelihood = 0
    gradient = np.zeros(
        shape=len(mu_hat.reshape(-1)) + len(gamma.reshape(-1)) + len(beta.reshape(-1))
    )
    for t in range(data.shape[1]):
        prev = log_likelihood
        ext = S[t]

        nf = return_Nf(t, f, theta, data)
        xf = return_Xf(t, f, theta, X)
        lambda_indpt = return_lambda_indpt(alpha, mu, mu_hat, nf, ext)

        T = return_T(t, gamma, beta, lambda_indpt, xf, N_mean, N_scale, X_mean, X_scale)
        A = np.exp((T.transpose() - T.max(axis=1)).transpose())
        s = (np.divide(A.transpose(), A.sum(axis=1))).transpose()
        lambda_total = lambda_indpt.sum(axis=1)
        lambda_pi = (lambda_total * s.transpose()).transpose()

        for p in range(P):
            lambda_p_grad = return_gradient_of_lambda_p(
                p, t, mu_hat, gamma, beta, nf, ext, data, alpha, mu
            )

            for i in range(M):
                log_likelihood += (
                    data[p, t, i] * (np.log(lambda_total[p]) + np.log(s[p, i]))
                    - lambda_pi[p, i]
                ) * (1 / ll_scaling[p])

                mod_part = np.sum(
                    [
                        (np.int(i == j) - s[p, j])
                        * return_gradient_of_T_p_i(
                            j,
                            p,
                            t,
                            mu_hat,
                            beta,
                            gamma,
                            xf,
                            lambda_indpt,
                            mu,
                            ext,
                            N_mean,
                            N_scale,
                            X_mean,
                            X_scale,
                        )
                        for j in range(M)
                    ],
                    axis=0,
                )
                modulator = (lambda_p_grad / (lambda_total[p] + eps)) + mod_part
                diff = data[p, t, i] - lambda_pi[p, i]
                gradient += (modulator * diff) * (1 / ll_scaling[p])

    if time_averaged:
        T = data.shape[1]
        log_likelihood *= -1 / T
        gradient *= -1 / T
    else:
        log_likelihood *= -1

    log_likelihood += regularization * (
        np.sum(np.abs(gamma.reshape(-1))) + np.sum(np.abs(beta.reshape(-1)))
    )

    gradient += regularization * (
        np.hstack(
            [
                np.zeros(shape=len(mu_hat.reshape(-1))),
                np.sign(gamma.reshape(-1)),
                np.sign(beta.reshape(-1)),
            ]
        )
    )

    return log_likelihood, gradient


def return_likelihood_and_gradient_from_flattened(
    plist, data, X, S, alpha, mu, theta, regularization, time_averaged
):
    P = data.shape[0]
    M = data.shape[-1]
    K = int((len(plist) - (P * M + P * P * M * M)) / (P * M))

    mu_hat, gamma, beta = unflatten_parameters(plist, P, M, K)

    return return_likelihood_and_gradient(
        mu_hat, gamma, beta, data, X, S, alpha, mu, theta, regularization, time_averaged
    )


def get_likelihood_and_gradient_of_set(
    samples,
    X,
    S,
    x0,
    alpha,
    mu,
    theta,
    regularization,
    time_averaged,
    how_long,
    how_many_samples,
):

    nlls = []
    grads = []
    for i in range(how_many_samples):
        nll, grad = return_likelihood_and_gradient_from_flattened(
            x0,
            samples[i, :, :how_long, :],
            X[:, :how_long],
            S[:how_long],
            alpha,
            mu,
            theta,
            regularization,
            time_averaged,
        )
        nlls.append(nll)
        grads.append(grad)

    return np.sum(nlls, axis=0), np.sum(grads, axis=0)


class MARKETSHAREMODELWITHSAMPLES(object):
    # model object to fitted as input to IPOPT

    def __init__(
        self,
        p0,
        samples,
        X,
        S,
        alpha,
        mu,
        theta,
        regularization,
        time_averaged,
        P,
        M,
        K,
        how_long,
        how_many_samples,
        logfit_label,
    ):
        self.M = M
        self.P = P
        self.K = K
        self.samples = samples
        self.X = X
        self.S = S
        self.alpha = alpha
        self.mu = mu
        self.theta = theta
        self.start_time = perf_counter()
        self.regularization = regularization
        self.time_averaged = time_averaged
        self.how_long = how_long
        self.how_many_samples = how_many_samples

        _, grad = get_likelihood_and_gradient_of_set(
            self.samples,
            self.X,
            self.S,
            p0,
            self.alpha,
            self.mu,
            self.theta,
            self.regularization,
            self.time_averaged,
            self.how_long,
            self.how_many_samples,
        )
        self.grad = grad

        logging.basicConfig(
            filename=f"log/opt2:{logfit_label}.log",
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            force=True,
        )

    def objective(self, plist):
        start = perf_counter()

        objective, gradient = get_likelihood_and_gradient_of_set(
            self.samples,
            self.X,
            self.S,
            plist,
            self.alpha,
            self.mu,
            self.theta,
            self.regularization,
            self.time_averaged,
            self.how_long,
            self.how_many_samples,
        )
        self.grad = gradient

        logging.info(f"\tobj and grad eval took {perf_counter()-start}s.")
        logging.info(f"\tCurrent pt: {str(plist)}")
        # print("OBJECTIVE", objective)
        return objective

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        return self.grad

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        return np.array(
            [np.sum([x[p * self.M : (p + 1) * self.M]]) for p in range(self.P)]
        )

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        return None

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):

        logging.info(
            f"Objective value at iteration #{iter_count} is - {obj_value}. Running time: {int(perf_counter() - self.start_time)}s"
        )
        logging.info(f"gradient: {self.grad}")

        print(
            f"Objective value at iteration #{iter_count} is - {obj_value}. Running time: {int(perf_counter() - self.start_time)}s"
        )


def run_optimization_many_sequences_given_starting_point(
    p0,
    samples,
    X,
    S,
    alpha,
    mu,
    theta,
    regularization,
    time_averaged,
    P,
    M,
    K,
    how_long,
    how_many_samples,
    logfit_label,
):
    time_start = perf_counter()

    lb = [0] * (P * M) + [-1] * (P * M * K) + [-1] * (P * P * M * M)
    ub = [1] * (P * M) + [1] * (P * M * K) + [1] * (P * P * M * M)

    cl = [1] * P
    cu = [1] * P

    nlp = ipopt.problem(
        n=len(p0),
        m=len(cl),
        problem_obj=MARKETSHAREMODELWITHSAMPLES(
            p0,
            samples,
            X,
            S,
            alpha,
            mu,
            theta,
            regularization,
            time_averaged,
            P,
            M,
            K,
            how_long,
            how_many_samples,
            logfit_label,
        ),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )
    nlp.addOption("mu_strategy", "adaptive")
    # nlp.addOption("tol", 0.1)
    nlp.addOption("jacobian_approximation", "finite-difference-values")
    nlp.addOption("hessian_approximation", "limited-memory")
    # nlp.addOption("gamma_theta", 0.01)
    nlp.addOption("acceptable_tol", 1e-1)
    nlp.addOption("acceptable_obj_change_tol", 1e-1)
    nlp.addOption("acceptable_compl_inf_tol", 100.0)
    nlp.addOption("acceptable_iter", 10)
    nlp.addOption("print_level", 0)
    nlp.addOption("max_iter", 1)
    # nlp.addOption("accept_after_max_steps", 2)

    x_opt, info = nlp.solve(p0)
    time_end = perf_counter()
    return [x_opt, info, time_end - time_start]


def return_random_starting_points(num_points, P, M, K):
    sp_list = []
    for k in range(num_points):
        prng = RandomState(10 + k)

        a, b = 0, 1
        gamma_bet_units = ((b - a) * prng.random(P * M * K) + a).reshape(P, M, K)
        for l in range(K):
            for m in range(M):
                if (l != m // 2) and (l != int((m // 2) + (K / 2))):
                    gamma_bet_units[:, m, l] = 0
        gamma_bet_units = gamma_bet_units.reshape(-1)
        beta_bet_units = (b - a) * prng.random(P * P * M * M) + a

        s0 = np.hstack([prng.random(P * M), gamma_bet_units, beta_bet_units])  # / 100

        sp_list.append(s0)
    return sp_list
