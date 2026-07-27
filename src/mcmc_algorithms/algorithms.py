import numpy as np
from scipy import stats
from tqdm import tqdm


# Accept-Reject algorithm

def accept_reject(r, pi, M, steps):
    # r is the proposal distribution, pi is the target distribution
    # M is the scaling factor 
    U = stats.uniform # uniform distribution

    accepted_xi = [] # list for all accepted xi values
    accepted_u = [] # list for all accepted u values

    for iter in range(steps): 
        xi = r.rvs() # next proposed xi
        u = U.rvs(0) # u is uniformly on [0,1]
        if u <= (pi.pdf(xi))/(M*r.pdf(xi)): # pdf is the probability density function
            accepted_u.append(u)
            accepted_xi.append(xi) # append if condition is true

    return accepted_xi # return the list of all accepted xis


#independent Metropolis-Hastings algorithm

def independent_MH(r, pi_r_ratio, steps):
    # r is the proposal distribution, pi_r_ratio is pi(x)/r(x)
    xi = 0 # start value for xi is 0
    accepted_xis = [] # list for all accepted xi values

    for iter in range(steps):
        xi_next = r.rvs() # next proposed xi
        u = stats.uniform.rvs() # u is uniformly on [0,1]
        if u <= pi_r_ratio(xi_next)/pi_r_ratio(xi):
            accepted_xis.append(xi_next) # append if condition is true
            xi = xi_next
        else: accepted_xis.append(xi)

    return accepted_xis # return the list of all accepted xis



#random walk Metropolis-Hastings algorithm

def random_walk_MH(random_walk, pi, var, steps):
    # random_walk is the proposed random walk with var as scale
    # pi is the target distribution
    xi = 0 # start value for xi is 0
    accepted_xis = [] # list for all accepted xi values

    for iter in range(steps):
        xi_next = random_walk.rvs(loc=xi, scale=var) # next proposed xi
        u = stats.uniform.rvs() # u is uniform on [0,1]
        if u <= pi(xi_next)/pi(xi):
            accepted_xis.append(xi_next) # append if condition is true
            xi = xi_next
        else: accepted_xis.append(xi)
    return accepted_xis # return the list of all accepted xis


# slice sampler
def target(x, alpha, rho):
    return np.exp(-rho*(x**2+alpha*np.exp(-x)))
# target as in example from thesis; values in example were alpha=5 and rho=1

def slice_sampler(target, alpha, rho, w, steps, start):
    # target is target distribution, alpha and rho values from example,
    u_list = [] # list for accepted u values
    x_list = [] # list for accepted x values
    x = start # x set to start value


    for i in range(steps):
        u = np.random.uniform(low=0, high= target(x,alpha,rho)) 
        # uniform between 0 and y value of target function

        # left and right bound adjustment in order to uniformly choose a 
        # new x value
        left_bound = x-w
        right_bound = x+w
        while target(left_bound,alpha,rho) > u:
            left_bound -= w 
        while target(right_bound, alpha, rho) > u:
            right_bound += w
        

        new_x = np.random.uniform(low=left_bound, high = right_bound)
        # choose a x at random between the bounds
        while target(new_x,alpha,rho)< u: # if new target(x) is smaller than u 
                                            # adjust bounds
            if new_x < x:
                left_bound = new_x
            else: right_bound = new_x
            new_x = np.random.uniform(low=left_bound, high = right_bound)

        x = new_x
        x_list.append(x)
        u_list.append(u)
    return x_list, u_list # return lists with all x and u values


# Gibbs sampler for an AR(1) stochastic volatility model, using the slice
# sampler to draw from each state's full conditional

def gibbs_sv_sampler(y, phi, sigma, beta, w, iterations, sampler_iterations):
    # y is the vector of observations, phi/sigma/beta are the AR(1) SV model
    # parameters, w is the slice sampler step-out width, iterations is the
    # number of Gibbs sweeps over all states, sampler_iterations is the
    # number of slice sampler draws taken per state per sweep

    n = len(y) - 1 # states are indexed 0..n
    x_vec = np.zeros(n + 1) # current point estimate for each state
    mu_vec = np.zeros(n + 1)
    alpha_vec = np.zeros(n + 1)
    rho_vec = np.zeros(n + 1)

    x_history = np.zeros((iterations, n + 1)) # x_vec after each sweep
    pi_hist = np.zeros((n + 1, sampler_iterations)) # last sweep's draws per state

    for it in tqdm(range(iterations)):
        for k in range(n + 1):
            if k == 0:
                mu_vec[0] = phi * x_vec[1] - sigma**2 / 2
                alpha_vec[0] = y[0]**2 * sigma**2 * np.exp(-mu_vec[0]) / beta**2
                rho_vec[0] = 1 / (2 * sigma**2)
            elif k == n:
                mu_vec[n] = phi * x_vec[n - 1] - sigma**2 / 2
                alpha_vec[n] = y[n]**2 * sigma**2 * np.exp(-mu_vec[n]) / beta**2
                rho_vec[n] = 1 / (2 * sigma**2)
            else:
                mu_vec[k] = (phi * (x_vec[k + 1] + x_vec[k - 1]) - sigma**2 / 2) / (1 + phi**2)
                alpha_vec[k] = y[k]**2 * sigma**2 * np.exp(-mu_vec[k]) / ((1 + phi**2) * beta**2)
                rho_vec[k] = (1 + phi**2) / (2 * sigma**2)

            pi_hist[k], _ = slice_sampler(target, alpha_vec[k], rho_vec[k], w, sampler_iterations, 0)

        # update each state to the mode of its full conditional draws
        for k in range(n + 1):
            counts, bins = np.histogram(pi_hist[k], bins=50, density=True)
            mode_index = counts.argmax()
            x_vec[k] = (bins[mode_index] + bins[mode_index + 1]) / 2

        x_history[it] = x_vec

    return x_history, pi_hist # per-sweep state history and last sweep's full-conditional draws