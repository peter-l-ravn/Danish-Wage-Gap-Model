import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d, interp_1d_vec
from consav.quadrature import log_normal_gauss_hermite

from optimizer import optimizer

from IPython.display import display, Math

class ModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass


    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 20
        par.T_shock = 10

        par.T_max = 200

        par.tol = 1e-6

        par.A =  5.0
        par.N_y =  1.9
        par.alpha =  0.90
        par.theta_l_y =  0.66
        par.theta_h_y =  5.0
        par.theta_l_o =  0.9
        par.theta_h_o =  4.7
        par.mu_y =  7.3
        par.mu_o =  None
        par.rho_h = 0.5
        par.rho_l =  0.8
        par.c =  0.5

        par.l_h_o_init = 0.2
        par.l_l_o_init = 0.45
        par.wage_h_o_init = 0.5
        par.wage_l_o_init = 0.4

        par.l_h_o_ss = np.inf
        par.l_l_o_ss = np.inf
        par.wage_h_o_ss = np.inf
        par.wage_l_o_ss = np.inf


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        sol.l_h_y = np.empty(par.T_max)
        sol.l_l_y = np.empty(par.T_max)
        sol.l_h_o = np.empty(par.T_max)
        sol.l_l_o = np.empty(par.T_max)

        sol.wage_h_y = np.empty(par.T_max)
        sol.wage_l_y = np.empty(par.T_max)
        sol.wage_h_o = np.empty(par.T_max)
        sol.wage_l_o = np.empty(par.T_max)

        sol.Y = np.empty(par.T_max)

        sol.K = np.empty(par.T_max)

        sol.c_bar = np.empty(par.T_max)

        sol.l_h_o[0] = par.l_h_o_init
        sol.l_l_o[0] = par.l_l_o_init
        sol.wage_h_o[0] = par.wage_h_o_init
        sol.wage_l_o[0] = par.wage_l_o_init

        sol.avg_wage_young = np.empty(par.T_max)
        sol.avg_wage_old = np.empty(par.T_max)

    def allocate_sim(self):
        """ allocate simulation """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        sim.l_h_y = np.empty(par.T)
        sim.l_l_y = np.empty(par.T)
        sim.l_h_o = np.empty(par.T)
        sim.l_l_o = np.empty(par.T)

        sim.wage_h_y = np.empty(par.T)
        sim.wage_l_y = np.empty(par.T)
        sim.wage_h_o = np.empty(par.T)
        sim.wage_l_o = np.empty(par.T)

        sim.Y = np.empty(par.T)

        sim.K = np.empty(par.T)

        sim.c_bar = np.empty(par.T)

        sim.l_h_o[0] = par.l_h_o_ss
        sim.l_l_o[0] = par.l_l_o_ss
        sim.wage_h_o[0] = par.wage_h_o_ss
        sim.wage_l_o[0] = par.wage_l_o_ss

        sim.avg_wage_young = np.empty(par.T)
        sim.avg_wage_old = np.empty(par.T)


    def solve(self, do_print=False):

        # a. unpack
        par = self.par
        sol = self.sol

        self.allocate()

        a = 0.0
        b = par.N_y

        t = 0
        eps = np.inf

        while t < (par.T_max - 1) and eps > par.tol:

            calc_equilibrium(par, sol, t, a, b, par.T_max, do_print=do_print)

            eps = abs(sol.wage_l_o[t+1] - sol.wage_l_o[t])

            par.l_h_o_ss = sol.l_h_o[t].copy()
            par.l_l_o_ss = sol.l_l_o[t].copy()
            par.wage_h_o_ss = sol.wage_h_o[t].copy()
            par.wage_l_o_ss = sol.wage_l_o[t].copy()

            t += 1

            if eps < par.tol:
                if do_print:
                    print(f"Convergence achieved at iteration {t} with eps = {eps:.2e}")

            if t == (par.T_max - 1):
                if do_print:
                    print(f"Maximum iterations reached without convergence. Final eps = {eps:.2e}")


    def simulate_par_shock(self, parameter_names, parameter_values, single_period_shock=False):

        par = self.par
        sim = self.sim

        self.allocate_sim()

        self._saved_par_values = {
                    name: getattr(par, name)
                    for name in parameter_names
                }

        a = 0.0
        b = par.N_y

        for t in range(par.T):

            
            if parameter_names == None:
                pass

            elif single_period_shock:
                if t == par.T_shock - 1:
                    for name, value in zip(parameter_names, parameter_values):
                        setattr(par, name, value)

                if t == par.T_shock:
                    for name, old_value in self._saved_par_values.items():
                        setattr(par, name, old_value)

            else:
                if t == par.T_shock - 1:
                    for name, value in zip(parameter_names, parameter_values):
                        setattr(par, name, value)

            
            calc_equilibrium(par, sim, t, a, b, par.T, do_print=False)

        for name, old_value in self._saved_par_values.items():
            setattr(par, name, old_value)


    def simulate_series_shock(self, series_names, series_values):

        par = self.par
        sim = self.sim

        self.allocate_sim()

        a = 0.0
        b = par.N_y
        

        for t in range(par.T):

            if series_names == None:
                pass

            else:
                if t == par.T_shock:
                    for name, value in zip(series_names, series_values):
                        getattr(sim, name)[t] = value
            
            # print(sim.l_h_y[t])

            calc_equilibrium(par, sim, t, a, b, par.T, do_print=False)



    def average_wage_change(self, print_components=False):
        par = self.par
        sim = self.sim

        self.allocate_sim()

        a = 0.0
        b = par.N_y

        t = 0

        sim.l_h_y[t] = optimizer(obj_function, a, b, args=(par, sim, t), tol=1e-6)

        sim.l_l_y[t] = par.N_y - sim.l_h_y[t]

        sim.K[t] = sim.l_h_o[t] + sim.l_h_y[t]

        Lh = func_Lh(par, sim, t)
        Ll = func_Ll(par, sim, t)

        sim.wage_h_y[t] = wage_h_y(par, dY_dLl(par, Lh, Ll))
        sim.wage_l_y[t] = wage_l_y(par, dY_dLl(par, Lh, Ll))

        sim.c_bar[t] = par.A*(par.theta_h_y - par.mu_y*par.theta_l_y)*d2Y_dLl_dLh(par, Ll, Lh)*par.theta_h_o

        sim.avg_wage_young[t] = sim.wage_l_y[t]*sim.l_l_y[t]/(sim.l_l_y[t] + sim.l_h_y[t]) + sim.wage_h_y[t]*sim.l_h_y[t]/(sim.l_l_y[t] + sim.l_h_y[t])
        sim.avg_wage_old[t] = sim.wage_l_o[t]*sim.l_l_o[t]/(sim.l_l_o[t] + sim.l_h_o[t]) + sim.wage_h_o[t]*sim.l_h_o[t]/(sim.l_l_o[t] + sim.l_h_o[t])


        # print(d2Y_dLl2(par, Ll, Lh)*(par.theta_h_y - par.theta_l_y)) # All correct sign here
        # print(dK_dlhy_prev(par, Ll, Lh) - par.rho_h) # All correct sign here
        # print(d2Y_dLl_dLh(par, Ll, Lh)*par.theta_h_o*par.rho_h) # All correct sign here

        return d_avg_wy_dhy(par, sim, t, Ll, Lh, print_components=print_components)


def calc_equilibrium(par, sol, t, a, b, T, do_print=False):


    sol.l_h_y[t] = optimizer(obj_function, a, b, args=(par, sol, t), tol=1e-6)

    sol.l_l_y[t] = par.N_y - sol.l_h_y[t]

    sol.K[t] = sol.l_h_o[t] + sol.l_h_y[t]

    Lh = func_Lh(par, sol, t)
    Ll = func_Ll(par, sol, t)

    sol.wage_h_y[t] = wage_h_y(par, dY_dLl(par, Ll, Lh))
    sol.wage_l_y[t] = wage_l_y(par, dY_dLl(par, Ll, Lh))

    sol.c_bar[t] = par.A*(par.theta_h_y - par.mu_y*par.theta_l_y)*d2Y_dLl_dLh(par, Ll, Lh)*par.theta_h_o

    sol.avg_wage_young[t] = sol.wage_l_y[t]*sol.l_l_y[t]/(sol.l_l_y[t] + sol.l_h_y[t]) + sol.wage_h_y[t]*sol.l_h_y[t]/(sol.l_l_y[t] + sol.l_h_y[t])
    sol.avg_wage_old[t] = sol.wage_l_o[t]*sol.l_l_o[t]/(sol.l_l_o[t] + sol.l_h_o[t]) + sol.wage_h_o[t]*sol.l_h_o[t]/(sol.l_l_o[t] + sol.l_h_o[t])

    constraints(par, sol, t, do_print=do_print)
    
    if t < T - 1:
        sol.l_h_o[t+1] = par.rho_h*sol.l_h_y[t]
        sol.l_l_o[t+1] = par.rho_l*sol.l_l_y[t]
    
        sol.wage_h_o[t+1] = sol.wage_h_y[t]
        sol.wage_l_o[t+1] = sol.wage_l_y[t]

        


def obj_function(l_h_y, par, sol, t):
    # diff = ((par.A*par.alpha*(par.theta_l_y*par.N_y + (par.theta_h_y - par.theta_l_y)*l_h_y)**(par.alpha - 1) \
    #         *func_Lh(par, sol, t)**(1-par.alpha)) / par.c) \
    #         *(par.theta_h_y - par.mu_y*par.theta_l_y) - sol.l_h_o[t] - l_h_y

    l_l_y = par.N_y - l_h_y

    Lh = par.theta_h_o * sol.l_h_o[t] + par.theta_h_y * l_h_y
    Ll = par.theta_l_o * sol.l_l_o[t] + par.theta_l_y * l_l_y

    K = sol.l_h_o[t] + l_h_y

    diff = (par.A/par.c) * (par.theta_h_y*dY_dLh(par, Ll, Lh) - par.mu_y*par.theta_l_y*dY_dLl(par, Ll, Lh)) - K
    
    return diff**2

def dY_dLl(par, Ll, Lh):
    return par.alpha*(Ll)**(par.alpha-1)*(Lh)**(1-par.alpha)

def d2Y_dLl2(par, Ll, Lh):
    return par.alpha*(par.alpha - 1)*(Ll)**(par.alpha - 2)*(Lh)**(1 - par.alpha)

def dY_dLh(par, Ll, Lh):
    return (1 - par.alpha)*(Ll)**(par.alpha)*(Lh)**(- par.alpha)

def d2Y_dLh2(par, Ll, Lh):
    return (-par.alpha)*(1 - par.alpha)*(Ll)**(par.alpha - 1)*(Lh)**(-par.alpha - 1)

def d2Y_dLl_dLh(par, Ll, Lh):
    return par.alpha*(1 - par.alpha)*(Ll)**(par.alpha - 1)*(Lh)**(-par.alpha)

def wage_l_y(par, dY_dLl):
    return par.A*par.theta_l_y*dY_dLl

def wage_h_y(par, dY_dLl):
    return par.mu_y*par.A*par.theta_l_y*dY_dLl

def func_Lh(par, sol, t):
    return par.theta_h_o*sol.l_h_o[t] + par.theta_h_y*sol.l_h_y[t]

def func_Ll(par, sol, t):
    return par.theta_l_o*sol.l_l_o[t] + par.theta_l_y*sol.l_l_y[t]

def d2Y_dLl2(par, Ll, Lh):
    return par.alpha * (par.alpha - 1) * Ll**(par.alpha - 2) * Lh**(1 - par.alpha)

def d2Y_dLh2(par, Ll, Lh):
    return -par.alpha * (1 - par.alpha) * Ll**par.alpha * Lh**(-par.alpha - 1)

def d2Y_dLl_dLh(par, Ll, Lh):
    return par.alpha * (1 - par.alpha) * Ll**(par.alpha - 1) * Lh**(-par.alpha)


def dK_dlhy_prev(par, Ll, Lh):
    M = (
        par.theta_h_y**2 * d2Y_dLh2(par, Ll, Lh)
        - (par.mu_y + 1) * par.theta_h_y * par.theta_l_y * d2Y_dLl_dLh(par, Ll, Lh)
        + par.mu_y * par.theta_l_y**2 * d2Y_dLl2(par, Ll, Lh)
    )

    N = (
        (par.mu_y + 1) * par.theta_h_y * par.theta_l_y * d2Y_dLl_dLh(par, Ll, Lh)
        - par.theta_h_o * par.mu_y * par.theta_l_y * d2Y_dLl_dLh(par, Ll, Lh)
        - par.mu_y * par.theta_l_y**2 * d2Y_dLl2(par, Ll, Lh)
        + (par.theta_h_o - par.theta_h_y) * par.theta_h_y * d2Y_dLh2(par, Ll, Lh)
    )

    return par.rho_h * par.A * N / (par.c - par.A * M)


def dLl_dlhy_prev(par, Ll, Lh):
    return (par.theta_h_y - par.theta_l_y) * (dK_dlhy_prev(par, Ll, Lh) - par.rho_h)


def dLh_dlhy_prev(par):
    return par.theta_h_o * par.rho_h


def dlhy_dlhy_prev(par, Ll, Lh):
    return dK_dlhy_prev(par, Ll, Lh) - par.rho_h


def dwLl_dlhy_prev(par, Ll, Lh):
    return par.A * par.theta_l_y * (
        d2Y_dLh2(par, Ll, Lh) * (par.theta_l_y * (par.rho_h - dK_dlhy_prev(par, Ll, Lh)) + d2Y_dLl_dLh(par, Ll, Lh) * (par.theta_h_y*dK_dlhy_prev(par, Ll, Lh) + par.rho_h*(par.theta_h_o - par.theta_h_y)))
    )


def d_avg_wy_dhy(par, sol, t, Ll, Lh, print_components=False):
    Ll = func_Ll(par, sol, t)
    Lh = func_Lh(par, sol, t)

    career = (par.mu_y - 1) / par.N_y * sol.wage_l_y[t] * dlhy_dlhy_prev(par, Ll, Lh)
    level = ((par.mu_y - 1) * sol.l_h_y[t] / par.N_y + 1) * dwLl_dlhy_prev(par, Ll, Lh)

    if print_components:
        print("Career spillovers", career)
        print("Wage level", level)

    return career + level, career, level






def constraints(par, sol, t, do_print=False):

    if par.theta_h_o < par.theta_l_o:
        if do_print:
            display(Math(r'\theta_h^o > \theta_{\ell}^o \text{ does not apply}'))
        return False

    if par.theta_h_y < par.theta_l_y:
        if do_print:
            display(Math(r'\theta_h^y > \theta_{\ell}^y \text{ does not apply}'))
        return False

    if par.mu_y < 1.0:
        if do_print:
            display(Math(r'\mu_y > 1 \text{ does not apply}'))
        return False

    if sol.wage_h_o[t] / sol.wage_l_o[t] < 1.0:
        if do_print:
            display(Math(r'\mu_y > 1\text{ does not apply}'))
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