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

import pandas as pd

class ModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass


    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.tol = 1e-6 # Convergence tolerance

        par.T_max = 200 # Max solver iterations

        par.N_rep = 200 # Number of represenatative agents
        par.N_first = 1 # Total mass of each cohort

        par.A =  1.0 # Total factor productivity
        par.alpha =  0.1 # Output elasticity of low-skilled labor
        par.c =  0.0 # Cost of hiring high-skilled labor

        par.gamma = 1.5
        par.delta = 0.05

        par.theta_mean = 0.0
        par.theta_std = 0.5

        # x = np.linspace(1.0, par.n, par.n)
        # rho_shape = 5.0
        # par.rho = -((x / par.n) ** rho_shape) + 1 # Cohort survival probabilities

        par.rho = 1 - pd.read_csv('Data/rho.csv', header=0)["rho"].values
        par.n = par.rho.shape[0] # Number of age cohorts


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

        shape = (par.T_max, par.n, par.N_rep)

        sol.wage = np.full(shape, np.nan)
        sol.wage_l = np.full(shape, np.nan)
        sol.wage_h = np.full(shape, np.nan)
        sol.l_h = np.full(shape, np.nan)
        sol.ability = np.full(shape, np.nan)
        sol.tenure = np.full(shape, np.nan)
        sol.theta_l = np.full(shape, np.nan)
        sol.theta_h = np.full(shape, np.nan)
        sol.mass = np.full(shape, np.nan)

        sol.profits = np.full((par.T_max), np.nan)
        sol.wage_sum_l = np.full((par.T_max), np.nan)
        sol.wage_sum_h = np.full((par.T_max), np.nan)


    def allocate_ss(self):

        # unpack
        par = self.par
        sol = self.sol

        shape = (par.n, par.N_rep, )

        sol.wage_ss = np.full(shape, np.nan)
        sol.wage_l_ss = np.full(shape, np.nan)
        sol.wage_h_ss = np.full(shape, np.nan)
        sol.l_h_ss = np.full(shape, np.nan)
        sol.ability_ss = np.full(shape, np.nan)
        sol.tenure_ss = np.full(shape, np.nan)
        sol.theta_l_ss = np.full(shape, np.nan)
        sol.theta_h_ss = np.full(shape, np.nan)
        sol.mass_ss = np.full(shape, np.nan)


    def init_fixed_draws(self):
        par = self.par
        sol = self.sol

        ability_draws, mass_draws = create_weighted_lognormal_distribution(
            par.theta_mean,
            par.theta_std,
            par.N_rep,
            total_mass=par.N_first,
        )
        sol.ability_draws = np.tile(ability_draws, (par.n, 1))
        sol.mass_draws = np.tile(mass_draws, (par.n, 1))

    def gen_first_period(self):
        
        par = self.par
        sol = self.sol

        mass = 1.0
        for age in range(par.n):

            sol.wage[0, age, :] = 1.0
            sol.wage_l[0, age, :] = 1.0
            sol.wage_h[0, age, :] = 1.0

            share_tenure = 0.5
            n_tenure = int(np.floor(par.N_rep * share_tenure))
            sol.tenure[0, age, :n_tenure] = 0.0
            sol.tenure[0, age, n_tenure:] = age

            sol.ability[0, age, :] = sol.ability_draws[age, :]
            sol.theta_l[0, age, :] = productivity_low(par, sol.ability[0, age, :], sol.tenure[0, age, :])
            sol.theta_h[0, age, :] = productivity_high(par, sol.ability[0, age, :], sol.tenure[0, age, :])
            sol.mass[0, age, :] = sol.mass_draws[age, :] * mass

            mass = mass * (1 - par.rho[age])


        sol.l_h[0, :, :] = 0.0

    def gen_first_period_from_ss(self):
        par = self.par
        sol = self.sol

        sol.wage[0, :, :] = sol.wage_ss[:]
        sol.wage_l[0, :, :] = sol.wage_l_ss[:]
        sol.wage_h[0, :, :] = sol.wage_h_ss[:]
        sol.l_h[0, :, :] = sol.l_h_ss[:]
        sol.ability[0, :, :] = sol.ability_ss[:]
        sol.tenure[0, :, :] = sol.tenure_ss[:]
        sol.theta_l[0, :, :] = sol.theta_l_ss[:]
        sol.theta_h[0, :, :] = sol.theta_h_ss[:]
        sol.mass[0, :, :] = sol.mass_ss[:]


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

            calc_equilibrium(par, sol, 0, do_print=do_print)

            for t in range(t_end + 1):
                if t > 0:
                    apply_retirement(par, sol, t)
                    calc_equilibrium(par, sol, t, do_print=do_print)

                if t < t_end:
                    law_of_motions(par, sol, t)





@jit_if_enabled()
def find_ss(par, sol, do_print=False):

    t = 0
    eps = np.inf

    while t < (par.T_max - 1) and eps > par.tol:

        apply_retirement(par, sol, t)
        calc_equilibrium(par, sol, t, do_print=do_print)

        law_of_motions(par, sol, t)

        if t == 0:
            eps = np.inf

        else:
            mass_change = np.max(
                np.abs(sol.mass[t] - sol.mass[t - 1])
            )

            # allocation_change = np.max(
            #     np.abs(sol.l_h[t] - sol.l_h[t - 1])
            # )

            tenure_change = np.max(
                np.abs(sol.tenure[t] - sol.tenure[t - 1])
            )

            eps = max(mass_change,  tenure_change)
            

            if not np.isfinite(eps):
                raise ValueError(
                    "Non-finite value encountered during steady-state iteration"
                )


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
            apply_retirement(par, sol, t)
            calc_equilibrium(par, sol, t, do_print=do_print)

            sol.wage_ss[:] = sol.wage[t]
            sol.wage_l_ss[:] = sol.wage_l[t]
            sol.wage_h_ss[:] = sol.wage_h[t]
            sol.l_h_ss[:] = sol.l_h[t]
            sol.ability_ss[:] = sol.ability[t]
            sol.tenure_ss[:] = sol.tenure[t]
            sol.theta_l_ss[:] = sol.theta_l[t]
            sol.theta_h_ss[:] = sol.theta_h[t]
            sol.mass_ss[:] = sol.mass[t]


@jit_if_enabled()
def calc_equilibrium(par, sol, t, do_print=False):

    sol.l_h[t] = high_skill_allocation(par, sol, t)

    Lh = np.nansum(sol.l_h[t] * sol.theta_h[t] * sol.mass[t])
    Ll = np.nansum((1.0 - sol.l_h[t]) * sol.theta_l[t] * sol.mass[t])

    marginal_product_high = dY_dLh(par, Ll, Lh)
    marginal_product_low = dY_dLl(par, Ll, Lh)

    sol.wage_h[t] = wage_h_func(par, sol, t, sol.theta_h[t], marginal_product_high)
    sol.wage_l[t] = wage_l_func(par, sol, t, sol.theta_l[t], marginal_product_low)
    sol.wage[t] = sol.l_h[t] * sol.wage_h[t] + (1.0 - sol.l_h[t]) * sol.wage_l[t]

    sol.wage_sum_h[t] = np.nansum(sol.l_h[t] * sol.wage_h[t] * sol.mass[t])
    sol.wage_sum_l[t] = np.nansum((1.0 - sol.l_h[t]) * sol.wage_l[t] * sol.mass[t])

    sol.Y[t] = par.A * Ll**par.alpha * Lh**(1.0 - par.alpha)
    sol.K[t] = np.nansum(sol.l_h[t] * sol.mass[t])
    sol.profits[t] = sol.Y[t] - sol.wage_sum_h[t] - sol.wage_sum_l[t] - (par.c / 2.0) * sol.K[t]**2


@jit_if_enabled()
def allocation_from_productivity_cutoff(cutoff, relative_wage_index, mass, order_by_age, tiny=1e-12):
    """Apply one common cutoff while interpolating the marginal ability bin separately within each age."""
    allocation = np.zeros(relative_wage_index.shape)
    
    for age in range(relative_wage_index.shape[0]):
        ordered_indices = order_by_age[age]
        active = np.isfinite(relative_wage_index[age, ordered_indices]) & np.isfinite(mass[age, ordered_indices]) & (mass[age, ordered_indices] > tiny)
        ordered_indices = ordered_indices[active]

        if ordered_indices.size == 0:
            continue

        scores = relative_wage_index[age, ordered_indices]

        if cutoff > scores[0]:
            continue

        if cutoff <= scores[-1]:
            allocation[age, ordered_indices] = 1.0
            continue

        upper_position = np.searchsorted(-scores, -cutoff, side="right") - 1
        lower_position = upper_position + 1

        allocation[age, ordered_indices[:upper_position + 1]] = 1.0

        score_distance = scores[upper_position] - scores[lower_position]

        fractional_share = (scores[upper_position] - cutoff) / score_distance if score_distance > tiny else 0.0

        allocation[age, ordered_indices[lower_position]] = np.clip(fractional_share, 0.0, 1.0)

    return allocation


@jit_if_enabled()
def productivity_cutoff_equation(cutoff, par, sol, t, relative_wage_index, mass, order_by_age):
    """Return the difference between the proposed productivity cutoff and the equilibrium wage-factor cutoff."""
    allocation = allocation_from_productivity_cutoff(cutoff, relative_wage_index, mass, order_by_age)

    Lh = np.nansum(allocation * sol.theta_h[t] * mass)
    Ll = np.nansum((1.0 - allocation) * sol.theta_l[t] * mass)

    marginal_product_high = dY_dLh(par, Ll, Lh)
    marginal_product_low = dY_dLl(par, Ll, Lh)

    unit = np.ones(1)

    high_wage_factor = float(wage_h_func(par, sol, t, unit, marginal_product_high)[0] / wage_h_func(par, sol, t, unit, 1.0)[0])
    low_wage_factor = float(wage_l_func(par, sol, t, unit, marginal_product_low)[0] / wage_l_func(par, sol, t, unit, 1.0)[0])

    return cutoff * high_wage_factor - low_wage_factor


@jit_if_enabled()
def high_skill_allocation(par, sol, t, do_print=False):

    tiny = 1e-12
    mass = sol.mass[t]
    active = np.isfinite(mass) & (mass > tiny) & np.isfinite(sol.theta_h[t]) & np.isfinite(sol.theta_l[t])

    if not np.any(active):
        return np.zeros(sol.theta_h[t].shape)

    # We initially rank everyone with aggregate marginal product normalized to 1, since it affect both groups equally.
    wage_h_index = wage_h_func(par, sol, t, sol.theta_h[t], np.ones(sol.theta_h[t].shape))
    wage_l_index = wage_l_func(par, sol, t, sol.theta_l[t], np.ones(sol.theta_l[t].shape))
    relative_wage_index = wage_h_index / np.maximum(wage_l_index, tiny)
    order_by_age = np.argsort(relative_wage_index, axis=1)[:, ::-1]

    a = float(np.min(relative_wage_index[active]))
    b = float(np.max(relative_wage_index[active]))

    optimizer_args = (par, sol, t, relative_wage_index, mass, order_by_age)

    f_a = productivity_cutoff_equation(a, *optimizer_args)
    f_b = productivity_cutoff_equation(b, *optimizer_args)

    if f_a == 0.0:
        cutoff = a

    elif f_b == 0.0:
        cutoff = b

    elif f_a * f_b < 0.0:
        cutoff = brentq(productivity_cutoff_equation, a, b, args=optimizer_args, xtol=par.tol)
        
    elif f_a > 0.0 and f_b > 0.0:
        cutoff = a

    else:
        return np.zeros(sol.theta_h[t].shape)
    
    return allocation_from_productivity_cutoff(cutoff, relative_wage_index, mass, order_by_age)


@jit_if_enabled()
def apply_retirement(par, sol, t):
    sol.mass[t] = sol.mass[t] * (1 - par.rho[:, np.newaxis])


@jit_if_enabled()
def law_of_motions(par, sol, t):


    sol.l_h[t + 1, 1:] = sol.l_h[t, :-1] # Old cohort retains their high-skilled labor status
    sol.l_h[t + 1, 0] = 0.0  # New cohort enters as low-skilled labor

    sol.ability[t + 1, 1:] = sol.ability[t, :-1] # Old cohort retains their ability
    sol.ability[t + 1, 0] = sol.ability_draws[0]  # New cohort draws new abilities

    representative_high_skill = sol.l_h[t, :-1] >= 1.0 - 1e-12
    sol.tenure[t + 1, 1:] = sol.tenure[t, :-1] + representative_high_skill # Only unsplit high-skilled bins update the single stored tenure; partial-bin mass is integrated but not propagated as one averaged worker
    sol.tenure[t + 1, 0] = 0.0  # New cohort starts with zero tenure2

    sol.theta_l[t + 1] = productivity_low(par, sol.ability[t + 1], sol.tenure[t + 1])
    sol.theta_h[t + 1] = productivity_high(par, sol.ability[t + 1], sol.tenure[t + 1])

    sol.mass[t + 1, 1:] = sol.mass[t, :-1] # Old cohort's mass adjusted by survival probability
    sol.mass[t + 1, 0] = sol.mass_draws[0]  # New cohort's mass is drawn from the lognormal distribution



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
def wage_l_func(par, sol, t, theta_l, dY_dLl_value):
    return np.ones_like(theta_l)



@jit_if_enabled()
def wage_h_func(par, sol, t, theta_h, dY_dLh_value):
    return par.A * theta_h * dY_dLh_value

@jit_if_enabled()
def func_Lh(par, sol, t):
    return np.nansum(sol.theta_h[t] * sol.l_h[t] * sol.mass[t])

@jit_if_enabled()
def func_Ll(par, sol, t):
    return np.nansum(sol.theta_l[t] * (1 - sol.l_h[t]) * sol.mass[t])





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



def productivity_low(par, ability, tenure):
    return ability   # Example: low-skilled productivity increases with ability and tenure

def productivity_high(par, ability, tenure):
    return ability + par.gamma * (1 - np.exp(-par.delta * tenure))  # Example: high-skilled productivity increases with ability and tenure
