from statistics import NormalDist

import numpy as np

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d, interp_1d_vec
from consav.quadrature import log_normal_gauss_hermite

from optimizers import golden, brentq, golden_section_int_modified, golden_section_modified

from IPython.display import display, Math

from EconModel import EconModelClass, jit
from numba import njit, prange
from jit_module import jit_if_enabled

class ModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass


    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.seed = 40

        par.wage_market = 'monopsony' # Options: 'competitive', 'monopsony'

        par.only_reassigned_are_hired = True # If True, only reassigned individuals are hired as high-skilled labor. If False, all individuals can be hired as high-skilled labor.

        par.tol = 1e-6 # Convergence tolerance
        par.optimizer_tol = 1e-4

        par.T = 10 # Periods to simulate
        par.T_max = 200 # Max solver iterations

        par.N_rep = 1000 # Number of represenatative agents
        par.N_1 = 20_000 # Total mass of each cohort
        par.n = 31 # Number of cohorts

        par.A =  400.0 # Total factor productivity
        par.alpha =  0.5 # Output elasticity of low-skilled labor
        par.mu =  1.1 # Wage premium for high-skilled labor
        par.c =  0.005 # Cost of hiring high-skilled labor

        # par.theta_l = np.loadtxt('Exogenous_estimation/theta_l.csv', delimiter=',')
        # par.theta_h =  np.loadtxt('Exogenous_estimation/theta_h.csv', delimiter=',')

        par.theta_l = np.loadtxt('Exogenous_estimation/theta_l.csv', delimiter=',')
        par.theta_h =  np.loadtxt('Exogenous_estimation/theta_h.csv', delimiter=',') 
        
        par.theta_mean = -0.0
        par.theta_std = 0.5

        x = np.linspace(1.0, par.n, par.n)
        rho_shape = 5.0
        par.rho = -((x / par.n) ** rho_shape) + 1 # Cohort survival probabilities

        par.tenure_param = 0.1

        par.reassigned_percentage = 0.05 # Percentage of high-skilled labor that can be fired each period


    def update_params(self):
        """ parameters to update iteratively """


    def allocate(self):
        """ allocate model """

        self.allocate_sol()
        self.allocate_ss()
        self.init_fixed_draws()
        

    def allocate_sol(self):

        # unpack
        par = self.par
        sol = self.sol

        sol.Y = np.full((par.T_max), np.nan)

        sol.K = np.full((par.T_max), np.nan)

        sol.c_bar = np.full((par.T_max), np.nan)

        max_capacity = int(par.N_rep * par.n)

        sol.age = np.full((max_capacity, par.T_max), -1, dtype=np.int64)
        sol.wage = np.full((max_capacity, par.T_max), np.nan)
        sol.wage_l = np.full((max_capacity, par.T_max), np.nan)
        sol.wage_h = np.full((max_capacity, par.T_max), np.nan)
        sol.l_h = np.full((max_capacity, par.T_max), np.nan)
        sol.ability = np.full((max_capacity, par.T_max), np.nan)
        sol.theta_l = np.full((max_capacity, par.T_max), np.nan)
        sol.theta_h = np.full((max_capacity, par.T_max), np.nan)
        sol.mass = np.full((max_capacity, par.T_max), np.nan)

        sol.profits = np.full((par.T_max), np.nan)
        sol.wage_sum_l = np.full((par.T_max), np.nan)
        sol.wage_sum_h = np.full((par.T_max), np.nan)
        sol.K = np.full((par.T_max), np.nan)

    def allocate_ss(self):

        # unpack
        par = self.par
        sol = self.sol

        max_capacity = int(par.N_rep * par.n)

        sol.age_ss = np.full((max_capacity), np.nan)
        sol.wage_ss = np.full((max_capacity), np.nan)
        sol.wage_l_ss = np.full(max_capacity, np.nan)
        sol.wage_h_ss = np.full(max_capacity, np.nan)
        sol.l_h_ss = np.full((max_capacity), np.nan)
        sol.ability_ss = np.full((max_capacity), np.nan)
        sol.theta_l_ss = np.full((max_capacity), np.nan)
        sol.theta_h_ss = np.full((max_capacity), np.nan)
        sol.mass_ss = np.full((max_capacity), np.nan)


    def init_fixed_draws(self):
        par = self.par
        sol = self.sol

        ability_draws, mass_draws = create_weighted_lognormal_distribution(
            par.theta_mean,
            par.theta_std,
            par.N_rep,
            total_mass=par.N_1,
        )
        sol.ability_draws = np.tile(ability_draws, par.n)
        sol.mass_draws = np.tile(mass_draws, par.n)

    def gen_first_period(self):
        
        par = self.par
        sol = self.sol

        start_idx = 0
        mass = 1.0
        for age in range(par.n):

            num_individuals = par.N_rep
                
            end_idx = start_idx + num_individuals

            sol.age[start_idx:end_idx, 0] = age
            sol.wage[start_idx:end_idx, 0] = 1.0
            sol.wage_l[start_idx:end_idx, 0] = 1.0
            sol.wage_h[start_idx:end_idx, 0] = 1.0

            sol.ability[start_idx:end_idx, 0] = sol.ability_draws[start_idx:end_idx]
            sol.theta_l[start_idx:end_idx, 0] = par.theta_l[age] + sol.ability[start_idx:end_idx, 0]
            sol.theta_h[start_idx:end_idx, 0] = par.theta_h[age] + sol.ability[start_idx:end_idx, 0]
            sol.mass[start_idx:end_idx, 0] = sol.mass_draws[start_idx:end_idx] * mass

            start_idx = end_idx

            mass = mass * par.rho[age]

        sol.l_h[:end_idx, 0] = reassign_func(par, costum_percentage = 0.02)

    def gen_first_period_from_ss(self):
        par = self.par
        sol = self.sol

        sol.age[:, 0] = sol.age_ss[:]
        sol.wage[:, 0] = sol.wage_ss[:]
        sol.wage_l[:, 0] = sol.wage_l_ss[:]
        sol.wage_h[:, 0] = sol.wage_h_ss[:]
        sol.l_h[:, 0] = sol.l_h_ss[:]
        sol.ability[:, 0] = sol.ability_ss[:]
        sol.theta_l[:, 0] = sol.theta_l_ss[:]
        sol.theta_h[:, 0] = sol.theta_h_ss[:]
        sol.mass[:, 0] = sol.mass_ss[:]


    def solve(self, do_print=False):

        # a. unpack
        
        with jit(self) as model:

            self.gen_first_period()

            par = model.par
            sol = model.sol

            find_ss(par, sol, do_print=do_print)


    def generate_transition(self, t_end, do_print=False):

        self.allocate_sol()
        self.gen_first_period_from_ss()

        with jit(self) as model:
            par = model.par
            sol = model.sol

            for t in range(t_end):
                law_of_motions(par, sol, t)
                calc_equilibrium(par, sol, t + 1, do_print=do_print)



@jit_if_enabled()
def find_ss(par, sol, do_print=False):

    t = 0
    eps = np.inf

    while t < (par.T_max - 1) and eps > par.tol:

        calc_equilibrium(par, sol, t, do_print=do_print)

        law_of_motions(par, sol, t)

        if t == 0:
            eps = 10e+10

        else:
            means_prev = group_means(sol.wage[:, t - 1], sol.age[:, t - 1])
            means_current = group_means(sol.wage[:, t], sol.age[:, t])
            eps = np.max(np.abs(means_prev - means_current))


        t += 1


        if do_print:
            print("Iteration: ", t, "eps = ", eps)

        if eps < par.tol:

            if do_print:
                print("Convergence achieved at iteration ", t, "with eps = ", eps)

        if t == (par.T_max - 1):
            
            if do_print:
                print("Maximum iterations reached without convergence. Final eps = ", eps)

        if t == par.T_max - 1 or eps < par.tol:
            calc_equilibrium(par, sol, t, do_print=do_print)

            sol.age_ss[:] = sol.age[:, t]
            sol.wage_ss[:] = sol.wage[:, t]
            sol.wage_l_ss[:] = sol.wage_l[:, t]
            sol.wage_h_ss[:] = sol.wage_h[:, t]
            sol.l_h_ss[:] = sol.l_h[:, t]
            sol.ability_ss[:] = sol.ability[:, t]
            sol.theta_l_ss[:] = sol.theta_l[:, t]
            sol.theta_h_ss[:] = sol.theta_h[:, t]
            sol.mass_ss[:] = sol.mass[:, t]


@jit_if_enabled()
def calc_equilibrium(par, sol, t, do_print=False):

    population_size = len(sol.age[:, t])

    reassigned_share = reassign_func(par)

    reassigned_mass = reassigned_share * sol.mass[:, t]
    retained_mass = sol.mass[:, t] - reassigned_mass

    qualification_sorted = np.argsort(sol.theta_h[:, t])[::-1]
    reassigned_mass_by_age = np.sum(reassigned_mass.reshape(par.n, par.N_rep), axis=1)

    a = 0.0
    b = float(len(qualification_sorted) - 1)
    optimizer_args = (par, sol, t, reassigned_mass, retained_mass, qualification_sorted, reassigned_mass_by_age)

    f_a = marginal_gain(a, *optimizer_args)
    f_b = marginal_gain(b, *optimizer_args)

    if f_a == 0.0:
        x_star = a
    elif f_b == 0.0:
        x_star = b
    elif f_a * f_b < 0.0:
        x_star = brentq(marginal_gain, a, b, args=optimizer_args, xtol=par.optimizer_tol)
    elif f_a > 0.0 and f_b > 0.0:
        x_star = b
    else:
        x_star = a

    x_star_floor = int(np.floor(x_star))
    x_star_share = x_star - x_star_floor
    x_star_ceil = min(x_star_floor + 1, len(qualification_sorted) - 1)

    high_floor, low_floor, Lh_floor, Ll_floor, K_floor = calc_Lh_Ll(
        par, sol, t, retained_mass, reassigned_mass, qualification_sorted, x_star_floor
    )

    high_ceil, low_ceil, Lh_ceil, Ll_ceil, K_ceil = calc_Lh_Ll(
        par, sol, t, retained_mass, reassigned_mass, qualification_sorted, x_star_ceil
    )

    high_mass = (1 - x_star_share) * high_floor + x_star_share * high_ceil
    low_mass = sol.mass[:, t] - high_mass

    Lh = (1 - x_star_share) * Lh_floor + x_star_share * Lh_ceil
    Ll = (1 - x_star_share) * Ll_floor + x_star_share * Ll_ceil
    K = np.nansum(high_mass)

    old_l_h = sol.l_h[:, t].copy()

    sol.l_h[:, t] = high_mass / sol.mass[:, t]

    if par.only_reassigned_are_hired:
        retained_high_mass = retained_mass * old_l_h
        hired_high_mass = high_mass - retained_high_mass
        retained_low_mass = retained_mass * (1.0 - old_l_h)
        reassigned_low_mass = reassigned_mass - hired_high_mass

    else:
        retained_high_mass = retained_mass * old_l_h
        hiring_pool_mass = sol.mass[:, t] - retained_high_mass
        hired_high_mass = high_mass - retained_high_mass
        hiring_pool_denominator = np.where(hiring_pool_mass > 0.0, hiring_pool_mass, 1.0)
        hire_share = np.where(hiring_pool_mass > 0.0, hired_high_mass / hiring_pool_denominator, 0.0)

        retained_low_pool = retained_mass * (1.0 - old_l_h)
        retained_low_mass = (1.0 - hire_share) * retained_low_pool
        reassigned_low_mass = (1.0 - hire_share) * reassigned_mass


    if par.wage_market == 'competitive':
        wage_h_target = wage_h(par, sol, t, dY_dLh(par, Ll, Lh))
    elif par.wage_market == 'monopsony':
        wage_h_target = wage_h(par, sol, t, dY_dLl(par, Ll, Lh))
    wage_l_target = wage_l(par, sol, t, dY_dLl(par, Ll, Lh))

    new_idx = slice(0, par.N_rep)
    old_idx = slice(par.N_rep, population_size)

    if t == 0:
        previous_wage_h = sol.wage_h[:population_size - par.N_rep, t].copy()
        previous_wage_l = sol.wage_l[:population_size - par.N_rep, t].copy()
    else:
        previous_wage_h = sol.wage_h[:population_size - par.N_rep, t - 1]
        previous_wage_l = sol.wage_l[:population_size - par.N_rep, t - 1]

    sol.wage_h[new_idx, t] = wage_h_target[new_idx]
    sol.wage_l[new_idx, t] = wage_l_target[new_idx]

    high_wage_bill = retained_high_mass[old_idx] * previous_wage_h + hired_high_mass[old_idx] * wage_h_target[old_idx]
    low_wage_bill = retained_low_mass[old_idx] * previous_wage_l + reassigned_low_mass[old_idx] * wage_l_target[old_idx]

    high_denominator = np.where(high_mass[old_idx] > 0.0, high_mass[old_idx], 1.0)
    low_denominator = np.where(low_mass[old_idx] > 0.0, low_mass[old_idx], 1.0)

    sol.wage_h[old_idx, t] = np.where(high_mass[old_idx] > 0.0, high_wage_bill / high_denominator, wage_h_target[old_idx])
    sol.wage_l[old_idx, t] = np.where(low_mass[old_idx] > 0.0, low_wage_bill / low_denominator, wage_l_target[old_idx])


    sol.wage[:, t] = (sol.l_h[:, t] * sol.wage_h[:, t] + (1.0 - sol.l_h[:, t]) * sol.wage_l[:, t])


    sol.wage_sum_h[t] = np.nansum(sol.wage_h[:, t] * high_mass)
    sol.wage_sum_l[t] = np.nansum(sol.wage_l[:, t] * low_mass)

    sol.Y[t] = par.A * Ll**par.alpha * Lh**(1.0 - par.alpha)

    sol.profits[t] = (
        sol.Y[t]
        - sol.wage_sum_h[t]
        - sol.wage_sum_l[t]
        - (par.c / 2.0) * K**2
    )

    sol.K[t] = K




@jit_if_enabled()
def calc_Lh_Ll(par, sol, t, retained_mass, reassigned_mass, qualification_sorted, qualified_idx):

    if par.only_reassigned_are_hired:
        promoted = qualification_sorted[:qualified_idx + 1]

        high_mass = retained_mass * sol.l_h[:, t]
        high_mass[promoted] += reassigned_mass[promoted]
        high_mass = np.clip(high_mass, 0.0, sol.mass[:, t])

        low_mass = sol.mass[:, t] - high_mass

        Lh = np.nansum(sol.theta_h[:, t] * high_mass)
        Ll = np.nansum(sol.theta_l[:, t] * low_mass)

        K = np.nansum(high_mass)

        return high_mass, low_mass, Lh, Ll, K

    else:
        promoted = qualification_sorted[:qualified_idx + 1]

        retained_high_mass = retained_mass * sol.l_h[:, t]
        hiring_pool_mass = sol.mass[:, t] - retained_high_mass

        high_mass = retained_high_mass.copy()
        high_mass[promoted] += hiring_pool_mass[promoted]
        high_mass = np.clip(high_mass, 0.0, sol.mass[:, t])

        low_mass = sol.mass[:, t] - high_mass

        Lh = np.nansum(sol.theta_h[:, t] * high_mass)
        Ll = np.nansum(sol.theta_l[:, t] * low_mass)
        K = np.nansum(high_mass)

        return high_mass, low_mass, Lh, Ll, K


@jit_if_enabled()
def marginal_gain(qualified_idx, par, sol, t, reassigned_mass, retained_mass, qualification_sorted, reassigned_mass_by_age):

    qualified_idx_floor = int(np.floor(qualified_idx))
    qualified_idx_share = qualified_idx - qualified_idx_floor
    qualified_idx_ceil = min(qualified_idx_floor + 1, len(qualification_sorted) - 1)

    high_floor, _, _, _, _ = calc_Lh_Ll(par, sol, t, retained_mass, reassigned_mass, qualification_sorted, qualified_idx_floor)
    high_ceil, _, _, _, _ = calc_Lh_Ll(par, sol, t, retained_mass, reassigned_mass, qualification_sorted, qualified_idx_ceil)

    high_mass = (1.0 - qualified_idx_share) * high_floor + qualified_idx_share * high_ceil
    low_mass = sol.mass[:, t] - high_mass

    Lh = np.nansum(sol.theta_h[:, t] * high_mass)
    Ll = np.nansum(sol.theta_l[:, t] * low_mass)
    K = np.nansum(high_mass)

    worker_floor = qualification_sorted[qualified_idx_floor]
    worker_ceil = qualification_sorted[qualified_idx_ceil]
    marginal_theta_h = (1.0 - qualified_idx_share) * sol.theta_h[worker_floor, t] + qualified_idx_share * sol.theta_h[worker_ceil, t]

    # The problem is that theta_l is arbritary given a specific age-cutoff, hence we need to estimate the marginal theta_l 
    # at all ages around the cutoff
    ability_at_cutoff = marginal_theta_h - par.theta_h[:par.n]
    ability_safe = np.maximum(ability_at_cutoff, 1e-12)
    standardized_ability = (np.log(ability_safe) - par.theta_mean) / par.theta_std
    ability_density = np.exp(-0.5 * standardized_ability**2) / (ability_safe * par.theta_std * np.sqrt(2.0 * np.pi))
    ability_density = np.where(ability_at_cutoff > 0.0, ability_density, 0.0)

    marginal_weights = reassigned_mass_by_age * ability_density
    total_marginal_weight = np.sum(marginal_weights)
    theta_l_at_cutoff = par.theta_l[:par.n] + ability_at_cutoff
    marginal_theta_l = np.sum(marginal_weights * theta_l_at_cutoff) / total_marginal_weight

    diff = (par.A / par.c) * (marginal_theta_h * dY_dLh(par, Ll, Lh) - par.mu * marginal_theta_l * dY_dLl(par, Ll, Lh)) - K

    return diff

@jit_if_enabled()
def law_of_motions(par, sol, t):

    sol.age[:, t + 1] = sol.age[:, t]

    sol.l_h[par.N_rep:, t + 1] = sol.l_h[:-par.N_rep, t] # Old cohort retains their high-skilled labor status
    sol.l_h[:par.N_rep, t + 1] = np.repeat(False, par.N_rep)  # New cohort enters as low-skilled labor

    sol.ability[par.N_rep:, t + 1] = sol.ability[:-par.N_rep, t] # Old cohort retains their ability
    sol.ability[:par.N_rep, t + 1] = sol.ability_draws[:par.N_rep]  # New cohort draws new abilities

    sol.theta_l[par.N_rep:, t + 1] = par.theta_l[sol.age[par.N_rep:, t + 1]] + sol.ability[par.N_rep:, t + 1] 
    sol.theta_l[:par.N_rep, t + 1] = par.theta_l[0] + sol.ability[:par.N_rep, t + 1]  # New cohort uses the first age's theta_l

    sol.theta_h[par.N_rep:, t + 1] = par.theta_h[sol.age[par.N_rep:, t + 1]] + sol.ability[par.N_rep:, t + 1]
    sol.theta_h[:par.N_rep, t + 1] = par.theta_h[0] + sol.ability[:par.N_rep, t + 1]  # New cohort uses the first age's theta_h

    sol.mass[par.N_rep:, t + 1] = sol.mass[:-par.N_rep, t] * par.rho[sol.age[:-par.N_rep, t]] # Old cohort's mass adjusted by survival probability
    sol.mass[:par.N_rep, t + 1] = sol.mass_draws[:par.N_rep]  # New cohort's mass is drawn from the lognormal distribution



@jit_if_enabled()
def dY_dLl(par, Ll, Lh):
    Ll_safe = np.maximum(Ll, 1e-12)
    Lh_safe = np.maximum(Lh, 1e-12)
    return par.alpha * Ll_safe**(par.alpha - 1) * Lh_safe**(1 - par.alpha)

@jit_if_enabled()
def d2Y_dLl2(par, Ll, Lh):
    return par.alpha*(par.alpha - 1)*(Ll)**(par.alpha - 2)*(Lh)**(1 - par.alpha)

@jit_if_enabled()
def dY_dLh(par, Ll, Lh):
    Ll_safe = np.maximum(Ll, 1e-12)
    Lh_safe = np.maximum(Lh, 1e-12)
    return (1 - par.alpha) * Ll_safe**par.alpha * Lh_safe**(-par.alpha)

@jit_if_enabled()
def d2Y_dLh2(par, Ll, Lh):
    return (-par.alpha)*(1 - par.alpha)*(Ll)**(par.alpha)*(Lh)**(-par.alpha - 1)

@jit_if_enabled()
def d2Y_dLl_dLh(par, Ll, Lh):
    return par.alpha*(1 - par.alpha)*(Ll)**(par.alpha - 1)*(Lh)**(-par.alpha)

@jit_if_enabled()
def wage_l(par, sol, t, dY_dLl):
    return par.A*sol.theta_l[:, t]*dY_dLl

# @jit_if_enabled()
# def wage_h(par, sol, t, dY_dLh):
#     return par.A*sol.theta_h[:, t]*dY_dLh

@jit_if_enabled()
def wage_h(par, sol, t, dY_dX):
    if par.wage_market == 'competitive':
        return par.A*sol.theta_h[:, t]*dY_dX
    elif par.wage_market == 'monopsony':
        return par.mu*wage_l(par, sol, t, dY_dX) # Individuals earn a markup of the low-skilled wage based on the parameter mu

@jit_if_enabled()
def func_Lh(par, sol, t):
    return np.nansum(sol.theta_h[:, t] * sol.l_h[:, t] * sol.mass[:, t])

@jit_if_enabled()
def func_Ll(par, sol, t):
    return np.nansum(sol.theta_l[:, t] * (1 - sol.l_h[:, t]) * sol.mass[:, t])


@jit_if_enabled()
def reassign_func(par, costum_percentage = -1.0):
    if costum_percentage != -1:
        reassign_percentage = costum_percentage

    else:
        reassign_percentage = par.reassigned_percentage

    return reassign_percentage


@jit_if_enabled()
def group_means(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]
    b = b[mask]

    groups = np.unique(b)
    means = np.array([a[b == g].mean() for g in groups])

    return means



def create_weighted_lognormal_distribution(mean, sigma, n_obs, total_mass=1.0):
    """Approximate a lognormal distribution by equiprobable weighted points."""

    if n_obs < 1:
        raise ValueError("n_obs must be at least 1")

    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    weights = np.full(n_obs, total_mass / n_obs)

    if sigma == 0:
        return np.full(n_obs, np.exp(mean)), weights

    normal_dist = NormalDist(mu=mean, sigma=sigma)
    probabilities = (np.arange(n_obs) + 0.5) / n_obs
    abilities = np.exp(np.array([normal_dist.inv_cdf(p) for p in probabilities]))

    return abilities, weights