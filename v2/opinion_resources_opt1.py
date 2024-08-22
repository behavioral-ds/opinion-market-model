import ipopt, logging
import numpy as np
import pickle
from time import perf_counter, sleep
from numpy.random import RandomState

eps = 1e-9

def flatten_parameters(mu, alpha):
    return np.hstack([mu.reshape(-1), alpha.reshape(-1), theta.reshape(-1)])

def unflatten_parameters(plist, P):
    mu = plist[:P]
    alpha = plist[P:P+P*P].reshape(P,P)
    theta = plist[P+P*P]
    return mu, alpha, theta

def f(t, theta):
    return theta * ((1 - theta) ** (t-1))

def return_derivative_of_f_wrt_theta(t, theta):
    return (1-theta)**(t-2) * (1-theta*t)

def return_Nf(t, f, theta, data):
    P = data.shape[0]
    if t == 0:
        return np.zeros(P)
    else:
        return np.sum([f(t-s, theta) * data[:,s,:] for s in range(t)], axis=0).sum(axis=1)

def return_dNf(t, f, theta, alpha, data):
    P = data.shape[0]
    if t == 0:
        return np.zeros(P)
    else:
        return np.dot(alpha, np.sum([return_derivative_of_f_wrt_theta(t-s, theta) * data[:,s,:] for s in range(t)], axis=0).sum(axis=1))
        

def return_lambda(alpha, mu, nf, ext):
    return mu*ext + np.dot(alpha, nf)

def return_gradient_of_lambda_p(
    p, 
    t, 
    alpha,
    mu,
    nf,
    dnf,
    ext,
    data
    ):
    
    # order is mu, alpha, gamma, beta
    
    P = data.shape[0]
    gradient = np.zeros(shape = P + P*P + 1)
    
    gradient[p] = ext
    
    for r in range(P):
        gradient[P + p*P+r] = nf[r]
        
    # theta
    gradient[P + P*P] = dnf[r]
        
    return gradient

def return_likelihood_and_gradient(    
        alpha,
        mu,
        theta,
        S,
        data,
        regularization,
        time_averaged = True
    ):

    factor = np.array([1,1])
    
    P = alpha.shape[0]
    
    log_likelihood = 0
    gradient = np.zeros(shape = P + P*P + 1)
    
    overall_p0=[]
    overall_p1=[]
    
    lls = [[],[]]
    for t in range(data.shape[1]):
        prev = log_likelihood
        ext = S[t]

        nf = return_Nf(t, f, theta, data)
        dnf = return_dNf(t, f, theta, alpha, data)
        lambda_total = return_lambda(alpha, mu, nf, ext)
                
        # print(t, lambda_indpt)
        for p in range(P):
            n_pt = np.sum(data[p,t,:])
            # here we multiply by the scaling factor
            log_likelihood += (n_pt * np.log(lambda_total[p]) - lambda_total[p]) * (1/factor[p])
            # print(p, (n_pt * np.log(lambda_total[p]) - lambda_total[p]))
            
            if p==0:
                overall_p0.append((n_pt * np.log(lambda_total[p]) - lambda_total[p]))
            else:
                overall_p1.append((n_pt * np.log(lambda_total[p]) - lambda_total[p]))                
                
            # print(t, p, (n_pt * np.log(lambda_total[p]) - lambda_total[p]) * (1/factor[p]))
            
            lambda_p_grad = return_gradient_of_lambda_p(
                p, 
                t, 
                alpha,
                mu,
                nf,
                dnf,
                ext,
                data
            )

            modulator = (lambda_p_grad/(lambda_total[p] + eps))
            diff = n_pt - lambda_total[p]
            gradient += (modulator * diff) * (1/factor[p])

    if time_averaged:
        T = data.shape[1]
        log_likelihood *= (-1 / T)
        gradient *= (-1 / T)
    else:
        log_likelihood *= -1
    
    
    factor = np.sum(np.mean(data, axis=1), axis=1)
    reg=regularization * (
        np.sum((mu.reshape(-1) / factor)**2) +
        np.sum(np.abs(alpha.reshape(-1))**2)
    )

    log_likelihood += reg

    gradient += 2 * regularization * (
        np.hstack([
            (mu.reshape(-1) / factor) / factor,
            alpha.reshape(-1),
            np.zeros(shape=1)
        ])
    )
    
    return log_likelihood, gradient

def return_likelihood_and_gradient_from_flattened(plist, data, S, regularization, time_averaged):
    P = data.shape[0]
    
    mu, alpha, theta = unflatten_parameters(plist, P)

    return return_likelihood_and_gradient(    
        alpha,
        mu,
        theta,
        S,
        data,
        regularization,
        time_averaged
    )

def get_likelihood_and_gradient_of_set(samples, x0, S, regularization, time_averaged, how_long, how_many_samples):
    
    nlls = []
    grads = []
    for i in range(how_many_samples):
        nll, grad = return_likelihood_and_gradient_from_flattened(x0, samples[i,:,:how_long,:], S[:how_long], regularization, time_averaged)
        nlls.append(nll)
        grads.append(grad)
        
    return np.sum(nlls, axis=0), np.sum(grads, axis=0)

class MARKETSHAREMODELWITHSAMPLES(object):
    # model object to fitted as input to IPOPT

    def __init__(
        self,
        p0,
        S,
        samples,
        regularization,
        time_averaged,
        P,
        how_long,
        how_many_samples,
        logfit_label
    ):
        self.P = P
        self.samples = samples
        self.S = S
        self.start_time = perf_counter()
        self.regularization = regularization
        self.time_averaged = time_averaged
        self.how_long = how_long
        self.how_many_samples = how_many_samples
        
        obj, grad = get_likelihood_and_gradient_of_set(self.samples, p0, self.S, self.regularization, self.time_averaged, self.how_long, self.how_many_samples)
        self.grad = grad
        
        logging.basicConfig(filename=f"log/opt1:{logfit_label}.log", level=logging.INFO, format='%(asctime)s | %(message)s', force=True)
        logging.info(
            f"\tinitial obj:{obj}. \tinintial plist: {str(p0)}. initial grad: {self.grad}")

    def objective(self, plist):
        start = perf_counter()

        objective, gradient = get_likelihood_and_gradient_of_set(self.samples, plist, self.S, self.regularization, self.time_averaged, self.how_long, self.how_many_samples)
        self.grad = gradient

        logging.info(
            f"\tobj and grad eval took {perf_counter()-start}s."
        )
        logging.info(
            f"\tCurrent pt: {str(plist)}. \tCurrentobj: {objective}. \tCurrentgrad: {self.grad}"
        )
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
        return np.array([])
        
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
        logging.info(
            f"gradient: {self.grad}"
        )

        print(
            f"Objective value at iteration #{iter_count} is - {obj_value}. Running time: {int(perf_counter() - self.start_time)}s"
        )

def run_optimization_many_sequences_given_starting_point(
    p0,
    S,
    samples,
    regularization,
    time_averaged,
    P,
    how_long,
    how_many_samples,
    logfit_label
):
    time_start = perf_counter()

    lb = [0] * (P) \
        + [0] * (P * P + 1)
    ub = [1e6] * (P) \
        + [1, 1e6, 1e6, 1] \
        + [1]

    cl = []
    cu = []

    nlp = ipopt.problem(
        n=len(p0),
        m=len(cl),
        problem_obj=MARKETSHAREMODELWITHSAMPLES(
            p0,
            S,
            samples,
            regularization,
            time_averaged,
            P,
            how_long,
            how_many_samples,
            logfit_label
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
    # nlp.addOption("acceptable_compl_inf_tol", 100.)
    nlp.addOption("acceptable_iter", 10)
    nlp.addOption("print_level", 0)
    nlp.addOption("max_iter", 50)
    # nlp.addOption("accept_after_max_steps", 2)

    x_opt, info = nlp.solve(p0)
    time_end = perf_counter()
    return [x_opt, info, time_end - time_start]


def return_random_starting_points(num_points, samples, P):
    sp_list = []
    for k in range(num_points):
        prng = RandomState(10+k)
        s0 = np.hstack([
            prng.randint(0, P*np.max(samples)+1, size=P),
            prng.random(P*P),
            prng.random()
        ])
        sp_list.append(s0)
    return sp_list