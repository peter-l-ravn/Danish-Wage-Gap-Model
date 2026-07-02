from xml.parsers.expat import model

import numpy as np

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d, interp_1d_vec
from consav.quadrature import log_normal_gauss_hermite

from optimizers import golden, brentq, golden_section_int_modified

from IPython.display import display, Math

class ModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass


    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.tol = 1e-3 # Convergence tolerance

        par.T = 10 # Periods to simulate
        par.T_max = 100 # Max solver iterations

        par.N_1 = 10_000 # New entrants per cohort
        par.n = 31 # Number of cohorts

        par.A =  100.0 # Total factor productivity
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
        self.gen_first_period()


    def allocate_sol(self):

        # unpack
        par = self.par
        sol = self.sol

        sol.Y = np.full((par.T_max), np.nan)

        sol.K = np.full((par.T_max), np.nan)

        sol.c_bar = np.full((par.T_max), np.nan)

        max_capacity_factor = 1.05
        max_capacity = par.N_1

        for age in range(par.n):
            max_capacity += int(par.N_1 * par.rho[age] * max_capacity_factor)

        sol.age = np.full((max_capacity, par.T_max), np.nan)
        sol.wage = np.full((max_capacity, par.T_max), np.nan)
        sol.l_h = np.full((max_capacity, par.T_max), np.nan)
        sol.ability = np.full((max_capacity, par.T_max), np.nan)
        sol.theta_l = np.full((max_capacity, par.T_max), np.nan)
        sol.theta_h = np.full((max_capacity, par.T_max), np.nan)


    def gen_first_period(self):
        
        par = self.par
        sol = self.sol

        start_idx = 0
        for age in range(par.n):
            if age == 0:
                num_individuals = par.N_1

            else:
                num_individuals = int(par.rho[age - 1] * num_individuals)
                
            end_idx = start_idx + num_individuals

            sol.age[start_idx:end_idx, 0] = age
            sol.wage[start_idx:end_idx, 0] = 1.0
            sol.l_h[start_idx:end_idx, 0] = age < 3
            sol.ability[start_idx:end_idx, 0] = draw_fixed_ability(par, num_individuals)
            sol.theta_l[start_idx:end_idx, 0] = par.theta_l[age] + sol.ability[start_idx:end_idx, 0]
            sol.theta_h[start_idx:end_idx, 0] = par.theta_h[age] + sol.ability[start_idx:end_idx, 0]

            start_idx = end_idx



    def solve(self, do_print=False):

        # a. unpack
        par = self.par
        sol = self.sol

        t = 0
        eps = np.inf


        while t < (par.T_max - 1) and eps > par.tol:
    
            calc_equilibrium(par, sol, t, par.T_max, do_print=do_print)

            law_of_motions(par, sol, t)

            if t == 0:
                eps = 10e+10

            else:
                eps = np.mean

                means_prev = group_means(sol.wage[:, t - 1], sol.age[:, t - 1])
                means_current = group_means(sol.wage[:, t], sol.age[:, t])
                eps = np.max(np.abs(means_prev - means_current))

            t += 1


            if do_print:
                print(f"Iteration {t}: eps = {eps:.2e}")

            if eps < par.tol:
                if do_print:
                    print(f"Convergence achieved at iteration {t} with eps = {eps:.2e}")

            if t == (par.T_max - 2):
                
                if do_print:
                    print(f"Maximum iterations reached without convergence. Final eps = {eps:.2e}")

            if t == par.T_max - 2 or eps < par.tol:
                calc_equilibrium(par, sol, t, par.T_max)




def calc_equilibrium(par, sol, t, T, do_print=False):

    population_size = np.count_nonzero(~np.isnan(sol.age[:, t]))

    reassigned_mask = np.zeros_like(sol.age[:, t], dtype=bool)

    np.random.seed(42)   # Set a fixed seed for reproducibility

    for age in range(par.n):
        idx = np.where(sol.age[:, t] == age)[0]
        reassigned_number = int(len(idx) * par.reassigned_percentage)

        chosen = draw_fixed_chosen(idx, reassigned_number)
        reassigned_mask[idx] = False
        reassigned_mask[chosen] = True

    a = 0
    b = len(reassigned_mask[reassigned_mask]) - 1

    x_star, f_star = golden_section_int_modified(a, b, par, sol, t, marginal_gain)

    qualified_idx = int(x_star)

    idx = np.where(reassigned_mask)[0]
    qualification_sorted = idx[np.argsort(sol.theta_h[idx, t])[::-1]]

    promoted = qualification_sorted[:qualified_idx + 1] 
    not_promoted = qualification_sorted[qualified_idx + 1:]

    sol.l_h[promoted, t] = True # Promote the top qualified_idx individuals to high-skilled labor
    sol.l_h[not_promoted, t] = False # Demote the rest to low-skilled labor

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



def marginal_gain(qualified_idx, par, sol, t):

    reassigned_mask = np.zeros_like(sol.age[:, t], dtype=bool)

    np.random.seed(42)   # Set a fixed seed for reproducibility

    for age in range(par.n):
        idx = np.where(sol.age[:, t] == age)[0]
        reassigned_number = int(len(idx) * par.reassigned_percentage)

        chosen = draw_fixed_chosen(idx, reassigned_number)
        reassigned_mask[idx] = False
        reassigned_mask[chosen] = True

    idx = np.where(reassigned_mask)[0]

    qualification_sorted = idx[np.argsort(sol.theta_h[idx, t])[::-1]]

    last_promoted = qualification_sorted[qualified_idx]

    promoted = qualification_sorted[:qualified_idx + 1] 
    not_promoted = qualification_sorted[qualified_idx + 1:]

    sol.l_h[promoted, t] = True # Promote the top qualified_idx individuals to high-skilled labor
    sol.l_h[not_promoted, t] = False # Demote the rest to low-skilled labor

    Lh = func_Lh(par, sol, t)
    Ll = func_Ll(par, sol, t)

    K = np.nansum(sol.l_h[:, t])

    diff = (par.A/par.c) * (sol.theta_h[last_promoted, t]*dY_dLh(par, Ll, Lh) - par.mu*sol.theta_l[last_promoted, t]*dY_dLl(par, Ll, Lh)) - K  

    sol.l_h[reassigned_mask, t] = False # Reset the high-skilled labor status for all individuals

    return diff


def law_of_motions(par, sol, t):

    new_cohort_age = np.zeros(par.N_1)
    old_cohort_age = sol.age[:, t]

    alive_mask = np.zeros_like(old_cohort_age, dtype=bool)

    for age in range(0, par.n):
        idx = np.where(old_cohort_age == age)[0]
        alive_number = int(len(idx) * par.rho[age])

        alive_mask[idx[:alive_number]] = True
        alive_mask[idx[alive_number:]] = False


    old_cohort_age = old_cohort_age + 1

    generation = np.concatenate((new_cohort_age, old_cohort_age[alive_mask]))

    sol.age[:len(generation), t + 1] = generation

    new_cohort_l_h = np.repeat(False, par.N_1)
    old_cohort_l_h = sol.l_h[:, t]
    sol.l_h[:len(generation), t + 1] = np.concatenate((new_cohort_l_h, old_cohort_l_h[alive_mask]))

    new_cohort_ability = draw_fixed_ability(par, par.N_1)
    sol.ability[:len(generation), t + 1] = np.concatenate((new_cohort_ability, sol.ability[alive_mask, t]))

    new_cohort_theta_l = par.theta_l[0] + new_cohort_ability
    new_cohort_theta_h = par.theta_h[0] + new_cohort_ability

    old_idx = old_cohort_age[alive_mask].astype(int)

    old_cohort_theta_l = par.theta_l[old_idx] + sol.ability[alive_mask, t]
    old_cohort_theta_h = par.theta_h[old_idx] + sol.ability[alive_mask, t]

    sol.theta_l[:len(generation), t + 1] = np.concatenate((new_cohort_theta_l, old_cohort_theta_l))
    sol.theta_h[:len(generation), t + 1] = np.concatenate((new_cohort_theta_h, old_cohort_theta_h))




def dY_dLl(par, Ll, Lh):
    return par.alpha*(Ll)**(par.alpha-1)*(Lh)**(1-par.alpha)

def d2Y_dLl2(par, Ll, Lh):
    return par.alpha*(par.alpha - 1)*(Ll)**(par.alpha - 2)*(Lh)**(1 - par.alpha)

def dY_dLh(par, Ll, Lh):
    return (1 - par.alpha)*(Ll)**(par.alpha)*(Lh)**(- par.alpha)

def d2Y_dLh2(par, Ll, Lh):
    return (-par.alpha)*(1 - par.alpha)*(Ll)**(par.alpha)*(Lh)**(-par.alpha - 1)

def d2Y_dLl_dLh(par, Ll, Lh):
    return par.alpha*(1 - par.alpha)*(Ll)**(par.alpha - 1)*(Lh)**(-par.alpha)

def wage_l(par, sol, t, dY_dLl):
    valid = ~np.isnan(sol.age[:, t])

    return par.A*sol.theta_l[valid, t]*dY_dLl

def wage_h(par, sol, t, dY_dLl):
    valid = ~np.isnan(sol.age[:, t])

    return par.mu*par.A*sol.theta_l[valid, t]*dY_dLl

def func_Lh(par, sol, t):
    valid = ~np.isnan(sol.age[:, t])
    l_hs = sol.l_h[valid, t].astype(bool)

    theta_h_repeated = sol.theta_h[valid, t][l_hs]
        

    return sum(theta_h_repeated)

def func_Ll(par, sol, t):
    valid = ~np.isnan(sol.age[:, t])
    l_hs = sol.l_h[valid, t].astype(bool)

    theta_l_repeated = sol.theta_l[valid, t][~l_hs]

    return sum(theta_l_repeated)


def draw_fixed_ability(par, num_individuals):
    np.random.seed(42)  # Set a fixed seed for reproducibility
    
    return np.random.lognormal(par.theta_mean, par.theta_std, num_individuals)

def draw_fixed_chosen(idx, reassigned_number):

    return np.random.choice(idx, size=reassigned_number, replace=False)


def group_means(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]
    b = b[mask]

    groups = np.unique(b)
    means = np.array([a[b == g].mean() for g in groups])

    return means






def params_to_latex(par, filename="params.tex", prefix="par"):
    """
    Save all parameters in a namespace/object as LaTeX commands.
    """

    lines = []

    for key, value in vars(par).items():

        # handle None
        if value is None:
            value_str = "None"

        # handle numpy scalars
        elif isinstance(value, np.generic):
            value_str = f"{value.item()}"

        # handle floats
        elif isinstance(value, float):
            value_str = f"{value:.10g}"

        else:
            value_str = str(value)

        line = rf"\newcommand{{\{prefix}_{key}}}{{{value_str}}}"
        lines.append(line)

    latex_code = "\n".join(lines)

    with open(filename, "w") as f:
        f.write(latex_code)

    print(f"Saved LaTeX commands to: {filename}")

    return latex_code