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

        par.n = 30

        par.T_max = 2000

        par.tol = 1e-8

        par.A =  5.0
        par.N_1 =  1.9
        par.alpha =  0.5

        par.theta_l_max = 2
        par.theta_l =  np.linspace(1.0, par.theta_l_max, par.n)

        par.theta_h_max = 2.2
        par.theta_h =  np.linspace(1.0, par.theta_h_max, par.n)
        
        par.mu =  1.1

        x = np.linspace(1.0, par.n - 1, par.n - 1)
        par.rho_h_shape = 5.0
        par.rho_h = -((x / par.n) ** par.rho_h_shape) + 1

        par.rho_l_shape = 5.0
        par.rho_l = -((x / par.n) ** par.rho_l_shape) + 1

        par.c =  0.5

        par.l_h_init = np.ones(par.n)*par.N_1*0.5
        par.l_l_init = np.ones(par.n)*(par.N_1 - par.l_h_init[0])
        par.wage_h_init = np.ones(par.n)*0.5
        par.wage_l_init = np.ones(par.n)*0.4

        par.l_h_ss = np.nan
        par.l_l_ss = np.nan
        par.wage_h_ss = np.nan
        par.wage_l_ss = np.nan


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        sol.l_h = np.full((par.T_max, par.n), np.nan)
        sol.l_l = np.full((par.T_max, par.n), np.nan)

        sol.wage_h = np.full((par.T_max, par.n), np.nan)
        sol.wage_l = np.full((par.T_max, par.n), np.nan)

        sol.Y = np.full((par.T_max), np.nan)

        sol.K = np.full((par.T_max), np.nan)

        sol.c_bar = np.full((par.T_max), np.nan)

        sol.l_h[0, :] = par.l_h_init
        sol.l_l[0, :] = par.l_l_init
        sol.wage_h[0, :] = par.wage_h_init
        sol.wage_l[0, :] = par.wage_l_init

        sol.avg_wage = np.full((par.T_max, par.n), np.nan)

    def allocate_sim(self):
        """ allocate simulation """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        sim.l_h = np.full((par.T, par.n), np.nan)
        sim.l_l = np.full((par.T, par.n), np.nan)

        sim.wage_h = np.full((par.T, par.n), np.nan)
        sim.wage_l = np.full((par.T, par.n), np.nan)

        sim.Y = np.full((par.T), np.nan)

        sim.K = np.full((par.T), np.nan)

        sim.c_bar = np.full((par.T), np.nan)

        sim.l_h[0, :] = par.l_h_ss
        sim.l_l[0, :] = par.l_l_ss
        sim.wage_h[0, :] = par.wage_h_ss
        sim.wage_l[0, :] = par.wage_l_ss

        sim.avg_wage = np.empty((par.T, par.n))


    def solve(self, do_print=False):

        # a. unpack
        par = self.par
        sol = self.sol

        self.allocate()

        a = 0.0
        b = par.N_1

        t = 0
        eps = np.inf

        while t < (par.T_max - 1) and eps > par.tol:

            calc_equilibrium(par, sol, t, a, b, par.T_max, do_print=do_print)

            eps = abs(sum(sol.wage_l[t+1, 1:] - sol.wage_l[t, 1:]))

            par.l_h_ss = sol.l_h[t, :].copy()
            par.l_l_ss = sol.l_l[t, :].copy()
            par.wage_h_ss = sol.wage_h[t, :].copy()
            par.wage_l_ss = sol.wage_l[t, :].copy()
        
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
        b = par.N_1

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
        
            calc_equilibrium(par, sim, t, a, b, par.T, do_print=False)



    def average_wage_change(self):
        par = self.par
        sim = self.sim

        self.allocate_sim()

        a = 0.0
        b = par.N_1

        t = 0

        sim.l_h[t, 0] = optimizer(obj_function, a, b, args=(par, sim, t), tol=1e-6)

        sim.l_l[t, 0] = par.N_1 - sim.l_h[t, 0]

        sim.K[t] = sum(sim.l_h[t, :])

        Lh = func_Lh(par, sim, t)
        Ll = func_Ll(par, sim, t)

        sim.wage_h[t, 0] = wage_h(par, dY_dLl(par, Ll, Lh))
        sim.wage_l[t, 0] = wage_l(par, dY_dLl(par, Ll, Lh))

        sim.avg_wage[t, :] = sim.wage_l[t, :]*sim.l_l[t, :]/(sim.l_l[t, :] + sim.l_h[t, :]) + sim.wage_h[t, :]*sim.l_h[t, :]/(sim.l_l[t, :] + sim.l_h[t, :])


def calc_equilibrium(par, sol, t, a, b, T, do_print=False):


    sol.l_h[t, 0] = optimizer(obj_function, a, b, args=(par, sol, t), tol=1e-6)

    sol.l_l[t, 0] = par.N_1 - sol.l_h[t, 0]

    sol.K[t] = sum(sol.l_h[t, :])

    Lh = func_Lh(par, sol, t)
    Ll = func_Ll(par, sol, t)

    sol.wage_h[t, 0] = wage_h(par, dY_dLl(par, Ll, Lh))
    sol.wage_l[t, 0] = wage_l(par, dY_dLl(par, Ll, Lh))


    sol.avg_wage[t] = sol.wage_l[t, :]*sol.l_l[t, :]/(sol.l_l[t, :] + sol.l_h[t, :]) + sol.wage_h[t, :]*sol.l_h[t, :]/(sol.l_l[t, :] + sol.l_h[t, :])

    constraints(par, sol, t, do_print=do_print)

    
    if t < T - 1:
        sol.l_h[t+1, 1:] = par.rho_h*sol.l_h[t, :-1]
        sol.l_l[t+1, 1:] = par.rho_l*sol.l_l[t, :-1]
    
        sol.wage_h[t+1, 1:] = sol.wage_h[t, :-1]
        sol.wage_l[t+1, 1:] = sol.wage_l[t, :-1]

        


def obj_function(l_h_1, par, sol, t):

    sol.l_h[t, 0] = l_h_1
    sol.l_l[t, 0] = par.N_1 - sol.l_h[t, 0]

    Lh = func_Lh(par, sol, t)
    Ll = func_Ll(par, sol, t)

    K = sum(sol.l_h[t, :])

    diff = (par.A/par.c) * (par.theta_h[0]*dY_dLh(par, Ll, Lh) - par.mu*par.theta_l[0]*dY_dLl(par, Ll, Lh)) - K    

    return diff**2

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

def wage_l(par, dY_dLl):
    return par.A*par.theta_l[0]*dY_dLl

def wage_h(par, dY_dLl):
    return par.mu*par.A*par.theta_l[0]*dY_dLl

def func_Lh(par, sol, t):
    return sum(par.theta_h[:]*sol.l_h[t, :])

def func_Ll(par, sol, t):
    return sum(par.theta_l[:]*sol.l_l[t, :])

def d2Y_dLl2(par, Ll, Lh):
    return par.alpha * (par.alpha - 1) * Ll**(par.alpha - 2) * Lh**(1 - par.alpha)

def d2Y_dLh2(par, Ll, Lh):
    return -par.alpha * (1 - par.alpha) * Ll**par.alpha * Lh**(-par.alpha - 1)

def d2Y_dLl_dLh(par, Ll, Lh):
    return par.alpha * (1 - par.alpha) * Ll**(par.alpha - 1) * Lh**(-par.alpha)



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