import numpy as np

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d, interp_1d_vec
from consav.quadrature import log_normal_gauss_hermite

from optimizers import golden, brentq, golden_section_int_modified

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

        par.tol = 1e-2 # Convergence tolerance

        par.T = 10 # Periods to simulate
        par.T_max = 50 # Max solver iterations

        par.N_1 = 10_000 # New entrants per cohort
        par.n = 31 # Number of cohorts

        par.A =  200.0 # Total factor productivity
        par.alpha =  0.5 # Output elasticity of low-skilled labor
        par.mu =  1.1 # Wage premium for high-skilled labor
        par.c =  0.5 # Cost of hiring high-skilled labor

        par.theta_l = np.loadtxt('Exogenous_estimation/theta_l.csv', delimiter=',')
        par.theta_h =  np.loadtxt('Exogenous_estimation/theta_h.csv', delimiter=',')
        
        par.theta_mean = -0.0
        par.theta_std = 0.5

        x = np.linspace(1.0, par.n, par.n)
        rho_shape = 5.0
        par.rho = -((x / par.n) ** rho_shape) + 1 # Cohort survival probabilities

        par.tenure_param = 0.1

        par.reassigned_percentage = 0.20 # Percentage of high-skilled labor that can be fired each period


    def update_params(self):
        """ parameters to update iteratively """


    def allocate(self):
        """ allocate model """

        self.allocate_sol()
        self.init_fixed_draws()
        

    def allocate_sol(self):

        # unpack
        par = self.par
        sol = self.sol

        sol.Y = np.full((par.T_max), np.nan)

        sol.K = np.full((par.T_max), np.nan)

        sol.c_bar = np.full((par.T_max), np.nan)

        max_capacity = int(par.N_1 * par.n)

        sol.age = np.full((max_capacity, par.T_max), -1, dtype=np.int64)
        sol.wage = np.full((max_capacity, par.T_max), np.nan)
        sol.l_h = np.full((max_capacity, par.T_max), np.nan)
        sol.ability = np.full((max_capacity, par.T_max), np.nan)
        sol.theta_l = np.full((max_capacity, par.T_max), np.nan)
        sol.theta_h = np.full((max_capacity, par.T_max), np.nan)
        sol.mass = np.full((max_capacity, par.T_max), np.nan)

        sol.age_ss = np.full((max_capacity), np.nan)
        sol.wage_ss = np.full((max_capacity), np.nan)
        sol.l_h_ss = np.full((max_capacity), np.nan)
        sol.ability_ss = np.full((max_capacity), np.nan)
        sol.theta_l_ss = np.full((max_capacity), np.nan)
        sol.theta_h_ss = np.full((max_capacity), np.nan)
        sol.mass_ss = np.full((max_capacity), np.nan)


    def init_fixed_draws(self):
        par = self.par
        sol = self.sol

        rng_ability = np.random.default_rng(par.seed)
        rng_reassign = np.random.default_rng(par.seed + 1)

        # one fixed draw per potential individual slot
        sol.ability_draws = rng_ability.lognormal(
            mean=par.theta_mean,
            sigma=par.theta_std,
            size=sol.age.shape[0]
        )

        sol.reassign_priority = rng_reassign.random(sol.age.shape[0])

    def gen_first_period(self):
        
        par = self.par
        sol = self.sol

        start_idx = 0
        mass = 1.0
        for age in range(par.n):

            num_individuals = par.N_1
                
            end_idx = start_idx + num_individuals

            sol.age[start_idx:end_idx, 0] = age
            sol.wage[start_idx:end_idx, 0] = 1.0

            sol.ability[start_idx:end_idx, 0] = sol.ability_draws[start_idx:end_idx]
            sol.theta_l[start_idx:end_idx, 0] = par.theta_l[age] + sol.ability[start_idx:end_idx, 0]
            sol.theta_h[start_idx:end_idx, 0] = par.theta_h[age] + sol.ability[start_idx:end_idx, 0]
            sol.mass[start_idx:end_idx, 0] = mass

            start_idx = end_idx

            mass = mass * par.rho[age]

        sol.l_h[:end_idx, 0] = reassign_func(par, sol, 0, costum_percentage = 0.02)[:end_idx]




    def solve(self, do_print=False):

        # a. unpack
        
        with jit(self) as model:

            self.gen_first_period()

            par = model.par
            sol = model.sol

            find_ss(par, sol, do_print=do_print)

@jit_if_enabled()
def find_ss(par, sol, do_print=False):

    t = 0
    eps = np.inf

    while t < (par.T_max - 1) and eps > par.tol:

        calc_equilibrium(par, sol, t, par.T_max, do_print=do_print)

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
            calc_equilibrium(par, sol, t, par.T_max)

            sol.age_ss[:] = sol.age[:, t]
            sol.wage_ss[:] = sol.wage[:, t]
            sol.l_h_ss[:] = sol.l_h[:, t]
            sol.ability_ss[:] = sol.ability[:, t]
            sol.theta_l_ss[:] = sol.theta_l[:, t]
            sol.theta_h_ss[:] = sol.theta_h[:, t]


@jit_if_enabled()
def calc_equilibrium(par, sol, t, T, do_print=False):

    population_size = len(sol.age[:, t])

    reassigned_mask = reassign_func(par, sol, t)

    idx = np.where(reassigned_mask)[0]

    qualification_sorted = idx[np.argsort(sol.theta_h[idx, t])[::-1]]

    sol.l_h[reassigned_mask, t] = False # All reassigned indivduals enter the labor market as low-skilled labor

    a = 0
    b = len(reassigned_mask[reassigned_mask]) - 1

    x_star, f_star = golden_section_int_modified(a, b, marginal_gain, par, sol, t, reassigned_mask, qualification_sorted)

    marginal_gain(int(x_star), par, sol, t, reassigned_mask, qualification_sorted)

    Lh = func_Lh(par, sol, t)
    Ll = func_Ll(par, sol, t)

    wage_h_target = wage_h(par, sol, t, dY_dLl(par, Ll, Lh))
    wage_l_target = wage_l(par, sol, t, dY_dLl(par, Ll, Lh))

    sol.wage[:par.N_1, t] = (sol.l_h[:par.N_1, t])*wage_h_target[:par.N_1] \
                          + (1 - sol.l_h[:par.N_1, t])*wage_l_target[:par.N_1]
    
    old_cohort_idx = slice(par.N_1, population_size)

    if t == 0:
        sol.wage[old_cohort_idx, t] = sol.l_h[old_cohort_idx, t] * (~reassigned_mask[old_cohort_idx]*sol.wage[:(population_size - par.N_1), t] + reassigned_mask[old_cohort_idx]*wage_h_target[old_cohort_idx]) \
                              + (1 - sol.l_h[old_cohort_idx, t]) * (~reassigned_mask[old_cohort_idx]*sol.wage[:(population_size - par.N_1), t] + reassigned_mask[old_cohort_idx]*wage_l_target[old_cohort_idx])

    else:
        sol.wage[old_cohort_idx, t] = sol.l_h[old_cohort_idx, t] * (~reassigned_mask[old_cohort_idx]*sol.wage[:(population_size - par.N_1), t - 1] + reassigned_mask[old_cohort_idx]*wage_h_target[old_cohort_idx]) \
                              + (1 - sol.l_h[old_cohort_idx, t]) * (~reassigned_mask[old_cohort_idx]*sol.wage[:(population_size - par.N_1), t - 1] + reassigned_mask[old_cohort_idx]*wage_l_target[old_cohort_idx])



@jit_if_enabled()
def marginal_gain(qualified_idx, par, sol, t, reassigned_mask, qualification_sorted):

    sol.l_h[reassigned_mask, t] = False # Reset the high-skilled labor status for all individuals

    last_promoted = qualification_sorted[qualified_idx]

    promoted = qualification_sorted[:qualified_idx + 1] 
    not_promoted = qualification_sorted[qualified_idx + 1:]

    sol.l_h[promoted, t] = True # Promote the top qualified_idx individuals to high-skilled labor
    sol.l_h[not_promoted, t] = False # Demote the rest to low-skilled labor

    Lh = func_Lh(par, sol, t)
    Ll = func_Ll(par, sol, t)

    K = np.nansum(sol.l_h[:, t] * sol.mass[:, t]) 

    diff = (par.A/par.c) * (sol.theta_h[last_promoted, t]*dY_dLh(par, Ll, Lh) - par.mu*sol.theta_l[last_promoted, t]*dY_dLl(par, Ll, Lh)) - K  

    return diff

@jit_if_enabled()
def law_of_motions(par, sol, t):

    sol.age[:, t + 1] = sol.age[:, t]

    sol.l_h[par.N_1:, t + 1] = sol.l_h[:-par.N_1, t] # Old cohort retains their high-skilled labor status
    sol.l_h[:par.N_1, t + 1] = np.repeat(False, par.N_1)  # New cohort enters as low-skilled labor

    sol.ability[par.N_1:, t + 1] = sol.ability[:-par.N_1, t] # Old cohort retains their ability
    sol.ability[:par.N_1, t + 1] = sol.ability_draws[:par.N_1]  # New cohort draws new abilities

    sol.theta_l[par.N_1:, t + 1] = par.theta_l[sol.age[par.N_1:, t + 1]] + sol.ability[par.N_1:, t + 1] 
    sol.theta_l[:par.N_1, t + 1] = par.theta_l[0] + sol.ability[:par.N_1, t + 1]  # New cohort uses the first age's theta_l

    sol.theta_h[par.N_1:, t + 1] = par.theta_h[sol.age[par.N_1:, t + 1]] + sol.ability[par.N_1:, t + 1]
    sol.theta_h[:par.N_1, t + 1] = par.theta_h[0] + sol.ability[:par.N_1, t + 1]  # New cohort uses the first age's theta_h

    sol.mass[par.N_1:, t + 1] = sol.mass[:-par.N_1, t] * par.rho[sol.age[par.N_1:, t + 1]] # Old cohort's mass adjusted by survival probability
    sol.mass[:par.N_1, t + 1] = 1.0



@jit_if_enabled()
def dY_dLl(par, Ll, Lh):
    return par.alpha*(Ll)**(par.alpha-1)*(Lh)**(1-par.alpha)

@jit_if_enabled()
def d2Y_dLl2(par, Ll, Lh):
    return par.alpha*(par.alpha - 1)*(Ll)**(par.alpha - 2)*(Lh)**(1 - par.alpha)

@jit_if_enabled()
def dY_dLh(par, Ll, Lh):
    return (1 - par.alpha)*(Ll)**(par.alpha)*(Lh)**(- par.alpha)

@jit_if_enabled()
def d2Y_dLh2(par, Ll, Lh):
    return (-par.alpha)*(1 - par.alpha)*(Ll)**(par.alpha)*(Lh)**(-par.alpha - 1)

@jit_if_enabled()
def d2Y_dLl_dLh(par, Ll, Lh):
    return par.alpha*(1 - par.alpha)*(Ll)**(par.alpha - 1)*(Lh)**(-par.alpha)

@jit_if_enabled()
def wage_l(par, sol, t, dY_dLl):

    return par.A*sol.theta_l[:, t]*dY_dLl

@jit_if_enabled()
def wage_h(par, sol, t, dY_dLl):
    return par.mu*wage_l(par, sol, t, dY_dLl) # Individuals earn a markup of the low-skilled wage based on the parameter mu

@jit_if_enabled()
def func_Lh(par, sol, t):
    return np.nansum(sol.theta_h[:, t] * sol.l_h[:, t] * sol.mass[:, t])

@jit_if_enabled()
def func_Ll(par, sol, t):
    return np.nansum(sol.theta_l[:, t] * (1 - sol.l_h[:, t]) * sol.mass[:, t])


@jit_if_enabled()
def reassign_func(par, sol, t, costum_percentage = -1.0):
    reassigned_mask = np.zeros(sol.age.shape[0], dtype=np.bool_)

    for age in range(par.n):
        idx = np.where(sol.age[:, t] == age)[0]
        if costum_percentage != -1:
            reassigned_number = int(len(idx) * costum_percentage)
        else:
            reassigned_number = int(len(idx) * par.reassigned_percentage)
    
        chosen = idx[np.argsort(sol.reassign_priority[idx])[:reassigned_number]]
        reassigned_mask[idx] = False
        reassigned_mask[chosen] = True

    return reassigned_mask

@jit_if_enabled()
def group_means(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]
    b = b[mask]

    groups = np.unique(b)
    means = np.array([a[b == g].mean() for g in groups])

    return means



