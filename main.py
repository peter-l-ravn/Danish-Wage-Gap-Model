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

        Lo = func_Lo(par, sim, t)
        Ly = func_Ly(par, sim, t)

        sim.wage_h_y[t] = wage_h_y(par, dY_dLy(par, Ly, Lo))
        sim.wage_l_y[t] = wage_l_y(par, dY_dLy(par, Ly, Lo))

        sim.c_bar[t] = par.A*(par.theta_h_y - par.mu_y*par.theta_l_y)*d2Y_dLy_dLo(par, Ly, Lo)*par.theta_h_o

        sim.avg_wage_young[t] = sim.wage_l_y[t]*sim.l_l_y[t]/(sim.l_l_y[t] + sim.l_h_y[t]) + sim.wage_h_y[t]*sim.l_h_y[t]/(sim.l_l_y[t] + sim.l_h_y[t])
        sim.avg_wage_old[t] = sim.wage_l_o[t]*sim.l_l_o[t]/(sim.l_l_o[t] + sim.l_h_o[t]) + sim.wage_h_o[t]*sim.l_h_o[t]/(sim.l_l_o[t] + sim.l_h_o[t])


        # print(d2Y_dLy2(par, Ly, Lo)*(par.theta_h_y - par.theta_l_y)) # All correct sign here
        # print(dK_dlhy_prev(par, Ly, Lo) - par.rho_h) # All correct sign here
        # print(d2Y_dLy_dLo(par, Ly, Lo)*par.theta_h_o*par.rho_h) # All correct sign here

        return d_avg_wy_dhy(par, sim, t, Ly, Lo, print_components=print_components)


def calc_equilibrium(par, sol, t, a, b, T, do_print=False):

    sol.l_h_y[t] = optimizer(obj_function, a, b, args=(par, sol, t), tol=1e-6)

    sol.l_l_y[t] = par.N_y - sol.l_h_y[t]

    sol.K[t] = sol.l_h_o[t] + sol.l_h_y[t]

    Lo = func_Lo(par, sol, t)
    Ly = func_Ly(par, sol, t)

    sol.wage_h_y[t] = wage_h_y(par, dY_dLy(par, Ly, Lo))
    sol.wage_l_y[t] = wage_l_y(par, dY_dLy(par, Ly, Lo))

    sol.c_bar[t] = par.A*(par.theta_h_y - par.mu_y*par.theta_l_y)*d2Y_dLy_dLo(par, Ly, Lo)*par.theta_h_o

    sol.avg_wage_young[t] = sol.wage_l_y[t]*sol.l_l_y[t]/(sol.l_l_y[t] + sol.l_h_y[t]) + sol.wage_h_y[t]*sol.l_h_y[t]/(sol.l_l_y[t] + sol.l_h_y[t])
    sol.avg_wage_old[t] = sol.wage_l_o[t]*sol.l_l_o[t]/(sol.l_l_o[t] + sol.l_h_o[t]) + sol.wage_h_o[t]*sol.l_h_o[t]/(sol.l_l_o[t] + sol.l_h_o[t])

    constraints(par, sol, t, do_print=do_print)
    
    if t < T - 1:
        sol.l_h_o[t+1] = par.rho_h*sol.l_h_y[t]
        sol.l_l_o[t+1] = par.rho_l*sol.l_l_y[t]
    
        sol.wage_h_o[t+1] = sol.wage_h_y[t]
        sol.wage_l_o[t+1] = sol.wage_l_y[t]

        


def obj_function(l_h_y, par, sol, t):
    diff = ((par.A*par.alpha*(par.theta_l_y*par.N_y + (par.theta_h_y - par.theta_l_y)*l_h_y)**(par.alpha - 1) \
            *func_Lo(par, sol, t)**(1-par.alpha)) / par.c) \
            *(par.theta_h_y - par.mu_y*par.theta_l_y) - sol.l_h_o[t] - l_h_y
    
    return diff**2

def dY_dLy(par, Ly, Lo):
    return par.alpha*(Ly)**(par.alpha-1)*(Lo)**(1-par.alpha)

def d2Y_dLy2(par, Ly, Lo):
    return par.alpha*(par.alpha - 1)*(Ly)**(par.alpha - 2)*(Lo)**(1 - par.alpha)

def dY_dLo(par, Ly, Lo):
    return (1 - par.alpha)*(Ly)**(par.alpha)*(Lo)**(- par.alpha)

def d2Y_dLo2(par, Ly, Lo):
    return (-par.alpha)*(1 - par.alpha)*(Ly)**(par.alpha)*(Lo)**(-par.alpha - 1)

def d2Y_dLy_dLo(par, Ly, Lo):
    return par.alpha*(1 - par.alpha)*(Ly)**(par.alpha - 1)*(Lo)**(-par.alpha)

def wage_l_y(par, dY_dLy):
    return par.A*par.theta_l_y*dY_dLy

def wage_h_y(par, dY_dLy):
    return par.mu_y*par.A*par.theta_l_y*dY_dLy

def func_Lo(par, sol, t):
    return par.theta_h_o*sol.l_h_o[t] + par.theta_l_o*sol.l_l_o[t]

def func_Ly(par, sol, t):
    return par.theta_h_y*sol.l_h_y[t] + par.theta_l_y*sol.l_l_y[t]

def dK_dlhy_prev(par, Ly, Lo,):
    return par.A*(par.theta_h_y - par.mu_y*par.theta_l_y) \
                    / (par.c - par.A*(par.theta_h_y - par.mu_y*par.theta_l_y)*d2Y_dLy2(par, Ly, Lo)*(par.theta_h_y - par.theta_l_y)) \
                    * (d2Y_dLy_dLo(par, Ly, Lo)*par.theta_h_o - d2Y_dLy2(par, Ly, Lo)*(par.theta_h_y - par.theta_l_y))*par.rho_h

def dLy_dlhy_prev(par, Ly, Lo):
    return (par.theta_h_y - par.theta_l_y)*(dK_dlhy_prev(par, Ly, Lo) - par.rho_h)

def dLo_dlhy_prev(par):
    return par.theta_h_o*par.rho_h

def dlhy_dlhy_prev(par, Ly, Lo):
    return dK_dlhy_prev(par, Ly, Lo) - par.rho_h


def dwly_dlhy_prev(par, Ly, Lo):
    return par.A*par.theta_l_y*(d2Y_dLy2(par, Ly, Lo)*(par.theta_h_y - par.theta_l_y)*(dK_dlhy_prev(par, Ly, Lo) - par.rho_h) + d2Y_dLy_dLo(par, Ly, Lo)*par.theta_h_o*par.rho_h)

def d_avg_wy_dhy(par, sol, t, Ly, Lo, print_components=False):
    ly = sol.l_l_y[t] + sol.l_h_y[t]
    
    career = 1/ly*(par.mu_y - 1)*sol.wage_l_y[t]*dlhy_dlhy_prev(par, Ly, Lo)
    level = (1/ly*(par.mu_y - 1)*sol.l_h_y[t] + 1)*dwly_dlhy_prev(par, Ly, Lo)

    if print_components:
        print("Career spillovers", 1/ly*(par.mu_y - 1)*sol.wage_l_y[t]*dlhy_dlhy_prev(par, Ly, Lo))
        print("Wage level", (1/ly*(par.mu_y - 1)*sol.l_h_y[t] + 1)*dwly_dlhy_prev(par, Ly, Lo))

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