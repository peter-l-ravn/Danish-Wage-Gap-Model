from main_new_prod import dY_dLh
import numpy as np

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d, interp_1d_vec
from consav.quadrature import log_normal_gauss_hermite

from optimizers import golden, brentq, golden_section_minimize_integer

from IPython.display import display, Math

class ModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass


    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 20 # Periods to simulate
        par.T_max = 10 # Max solver iterations

        par.N_1 = 1000 # New entrants per cohort
        par.n = 50 # Number of cohorts

        par.A =  100.0 # Total factor productivity
        par.alpha =  0.5 # Output elasticity of low-skilled labor
        par.mu =  1.1 # Wage premium for high-skilled labor
        par.phi = 0.9 # Calvo parameter for wage adjustment
        par.c =  0.5 # Cost of hiring high-skilled labor

        par.theta_l = np.loadtxt('Exogenous_estimation/theta_l.csv', delimiter=',')
        par.theta_h =  np.loadtxt('Exogenous_estimation/theta_h.csv', delimiter=',')
        
        par.theta_mean = -0.0
        par.theta_std = 0.5

        x = np.linspace(1.0, par.n, par.n)
        rho_shape = 5.0
        par.rho = -((x / par.n) ** rho_shape) + 1 # Cohort survival probabilities

        par.tenure_param = 0.1


    def update_params(self):

        """ parameters to update iteratively """


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol

        sol.Y = np.full((par.T_max), np.nan)

        sol.K = np.full((par.T_max), np.nan)

        sol.c_bar = np.full((par.T_max), np.nan)

        max_capacity_factor = 1.1
        max_capacity = int(par.N_1 * par.n * max_capacity_factor)

        sol.age = np.full((max_capacity, par.T_max), np.nan)
        sol.wage = np.full((max_capacity, par.T_max), np.nan)
        sol.l_h = np.full((max_capacity, par.T_max), np.nan)
        sol.theta_l = np.full((max_capacity, par.T_max), np.nan)
        sol.theta_h = np.full((max_capacity, par.T_max), np.nan)

        sol.age[:par.N_1*par.n, 0] = np.repeat(np.arange(0, par.n, 1), par.N_1)
        sol.wage[:par.N_1*par.n, 0] = np.repeat(np.ones((1, par.n)), par.N_1)
        sol.l_h[:par.N_1*par.n, 0] = sol.age[:par.N_1*par.n, 0] > 50
        sol.l_h[par.N_1*par.n-2, 0] = True 

        ability = np.random.lognormal(par.theta_mean, par.theta_std, par.N_1 * par.n)

        sol.theta_l[:par.N_1*par.n, 0] = np.repeat(par.theta_l, par.N_1) + ability
        sol.theta_h[:par.N_1*par.n, 0] = np.repeat(par.theta_h, par.N_1) + ability

        
    def allocate_sim(self, T):
        """ allocate simulation """



    def solve(self, do_print=False):

        # a. unpack
        par = self.par
        sol = self.sol

        self.allocate()

        t = 0
        eps = np.inf

        for t in range(par.T_max - 1):

            print(t)

            calc_equilibrium(par, sol, t, par.T_max, do_print=do_print)



        # while t < (par.T_max - 1) and eps > par.tol:

        #     calc_equilibrium(par, sol, t, par.T_max, do_print=do_print)

        #     if t == 0:
        #         eps = 10e+10
        #     else:
        #         eps = max(abs(sol.wage_l[t - 1, :] - sol.wage_l[t, :]))

        #     par.l_h_ss = sol.l_h[t, :].copy()
        #     par.l_l_ss = sol.l_l[t, :].copy()
        #     par.wage_h_ss = sol.wage_h[t, :].copy()
        #     par.wage_l_ss = sol.wage_l[t, :].copy()
        
        #     t += 1

        #     if eps < par.tol:
        #         if do_print:
        #             print(f"Convergence achieved at iteration {t} with eps = {eps:.2e}")

        #     if t == (par.T_max - 1):
                
        #         if do_print:
        #             print(f"Maximum iterations reached without convergence. Final eps = {eps:.2e}")



        

def calc_equilibrium(par, sol, t, T, do_print=False):

    sol.l_h[:par.N_1, t] = np.repeat(False, par.N_1)

    ability = np.random.lognormal(par.theta_mean, par.theta_std, par.N_1)

    sol.theta_l[:par.N_1, t] = par.theta_l[0] + ability
    sol.theta_h[:par.N_1, t] = par.theta_h[0] + ability

    a = 0
    b = np.count_nonzero(~np.isnan(sol.age[:, t])) - 1

    x_star, f_star = golden_section_minimize_integer(par, sol, t, marginal_gain)

    qualified_idx = int(x_star)

    valid = ~np.isnan(sol.age[:, t])

    qualification_sorted = np.argsort(sol.theta_h[valid, t])[::-1]

    promoted = qualification_sorted[:qualified_idx + 1] 

    sol.l_h[promoted, t] = True # Promote the top qualified_idx individuals to high-skilled labor
    sol.l_h[~promoted, t] = False # Demote the rest to low-skilled labor

    Lh = func_Lh(par, sol, t)
    Ll = func_Ll(par, sol, t)

    wage_h_target = wage_h(par, sol, t, dY_dLl(par, Ll, Lh))
    wage_l_target = wage_l(par, sol, t, dY_dLl(par, Ll, Lh))

    sol.wage[sol.l_h[:par.N_1, t], t] = wage_h_target[:par.N_1]
    sol.wage[~sol.l_h[:par.N_1, t], t] = wage_l_target[:par.N_1]

    if t == 0:
        sol.wage[sol.l_h[par.N_1:, t], t] = par.phi*sol.wage_h[t, :-1] + (1 - par.phi)*wage_h_target[par.N_1:]
        sol.wage_l[t, 1:] = par.phi*sol.wage_l[t, :-1] + (1 - par.phi)*wage_l_target[1:]
     
    else:
        sol.wage_h[t, 1:] = par.phi*sol.wage_h[t - 1, :-1] + (1 - par.phi)*wage_h_target[1:]
        sol.wage_l[t, 1:] = par.phi*sol.wage_l[t - 1, :-1] + (1 - par.phi)*wage_l_target[1:]



def marginal_gain(qualified_idx, par, sol, t):

    valid = ~np.isnan(sol.age[:, t])

    qualification_sorted = np.argsort(sol.theta_h[valid, t])[::-1]

    promoted = qualification_sorted[:qualified_idx + 1] 

    sol.l_h[promoted, t] = True # Promote the top qualified_idx individuals to high-skilled labor
    sol.l_h[~promoted, t] = False # Demote the rest to low-skilled labor

    Lh = func_Lh(par, sol, t)
    Ll = func_Ll(par, sol, t)

    K = np.nansum(sol.l_h[:, t])

    diff = (par.A/par.c) * (sol.theta_h[qualified_idx, t]*dY_dLh(par, Ll, Lh) - par.mu*sol.theta_l[qualified_idx, t]*dY_dLl(par, Ll, Lh)) - K  

    sol.l_h[valid, t] = False # Reset the high-skilled labor status for all individuals

    return diff ** 2



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





def constraints(par, sol, t, do_print=False):

    if np.any(par.theta_h < par.theta_l):
        if do_print:
            display(Math(r'\theta_h > \theta_{\ell} \text{ does not apply for some cohorts}'))
        return False

    if par.mu < 1.0:
        if do_print:
            display(Math(r'\mu > 1 \text{ does not apply}'))
        return False

    if np.any(sol.wage_h[t, :] / sol.wage_l[t, :] < 1.0):
        if do_print:
            display(Math(r'\mu > 1\text{ does not apply}'))
        return False

    return True



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