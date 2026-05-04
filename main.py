import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d, interp_1d_vec
from consav.quadrature import log_normal_gauss_hermite

from optimizer import optimizer

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

        par.A = 1.0

        par.N_y = 1

        par.alpha = 0.6

        par.theta_l_y = 1.0
        par.theta_h_y = 1.4
        par.theta_l_o = 1.1
        par.theta_h_o = 1.6

        par.mu_y = 1.2
        par.mu_o = 1.2

        par.rho_h = 0.6
        par.rho_l = 0.5

        par.c = 0.7

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


    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol

        self.allocate()

        a = 0.0
        b = par.N_y

        t = 0
        eps = np.inf

        while t < par.T_max and eps > 1e-6:

            sol.l_h_y[t] = optimizer(obj_function, a, b, args=(par, sol, t), tol=1e-6)

            sol.l_l_y[t] = par.N_y - sol.l_h_y[t]

            sol.K[t] = sol.l_h_o[t] + sol.l_h_y[t]

            Lo = func_Lo(par, sol, t)
            Ly = func_Ly(par, sol, t)

            sol.wage_h_y[t] = wage_h_y(par, dYdLy(par, Ly, Lo))
            sol.wage_l_y[t] = wage_l_y(par, dYdLy(par, Ly, Lo))

            sol.c_bar[t] = par.A*(par.theta_h_y - par.mu_y*par.theta_l_y)*d2YdLydLo(par, Ly, Lo)*par.theta_h_o


            if t < par.T_max - 1:
                sol.l_h_o[t+1] = par.rho_h*sol.l_h_y[t]
                sol.l_l_o[t+1] = par.rho_l*sol.l_l_y[t]
            
                sol.wage_h_o[t+1] = sol.wage_h_y[t]
                sol.wage_l_o[t+1] = sol.wage_l_y[t]

                eps = abs(sol.wage_l_o[t+1] - sol.wage_l_o[t])

            par.l_h_o_ss = sol.l_h_o[t]
            par.l_l_o_ss = sol.l_l_o[t]
            par.wage_h_o_ss = sol.wage_h_o[t]
            par.wage_l_o_ss = sol.wage_l_o[t]

            t += 1




    def simulate(self, parameter_names, parameter_values):

        par = self.par
        sim = self.sim

        self.allocate_sim()

        a = 0.0
        b = par.N_y

        for t in range(par.T):

            sim.l_h_y[t] = optimizer(obj_function, a, b, args=(par, sim, t), tol=1e-6)

            sim.l_l_y[t] = par.N_y - sim.l_h_y[t]

            sim.K[t] = sim.l_h_o[t] + sim.l_h_y[t]

            Lo = func_Lo(par, sim, t)
            Ly = func_Ly(par, sim, t)

            sim.wage_h_y[t] = wage_h_y(par, dYdLy(par, Ly, Lo))
            sim.wage_l_y[t] = wage_l_y(par, dYdLy(par, Ly, Lo))

            sim.c_bar[t] = par.A*(par.theta_h_y - par.mu_y*par.theta_l_y)*d2YdLydLo(par, Ly, Lo)*par.theta_h_o

            if t == par.T_shock - 1:
                for name, value in zip(parameter_names, parameter_values):
                    setattr(par, name, value)

            if t < par.T - 1:
                sim.l_h_o[t+1] = par.rho_h*sim.l_h_y[t]
                sim.l_l_o[t+1] = par.rho_l*sim.l_l_y[t]
            
                sim.wage_h_o[t+1] = sim.wage_h_y[t]
                sim.wage_l_o[t+1] = sim.wage_l_y[t]

        


def obj_function(l_h_y, par, sol, t):
    diff = ((par.A*par.alpha*(par.theta_l_y*par.N_y + (par.theta_h_y - par.theta_l_y)*l_h_y)**(par.alpha - 1) \
            *func_Lo(par, sol, t)**(1-par.alpha)) / par.c) \
            *(par.theta_h_y - par.mu_y*par.theta_l_y) - sol.l_h_o[t] - l_h_y
    
    return diff**2

def dYdLy(par, Ly, Lo):
    return par.alpha*(Ly)**(par.alpha-1)*(Lo)**(1-par.alpha)

def d2YdLydLy(par, Ly, Lo):
    return par.alpha*(par.alpha - 1)*(Ly)**(par.alpha - 2)*(Lo)**(1 - par.alpha)

def dYdLo(par, Ly, Lo):
    return (1 - par.alpha)*(Ly)**(par.alpha)*(Lo)**(- par.alpha)

def d2YdLodLo(par, Ly, Lo):
    return (-par.alpha)*(1 - par.alpha)*(Ly)**(par.alpha - 1)*(Lo)**(-par.alpha)

def d2YdLydLo(par, Ly, Lo):
    return par.alpha*(1 - par.alpha)*(Ly)**(par.alpha - 1)*(Lo)**(-par.alpha)

def wage_l_y(par, dYdLy):
    return par.A*par.theta_l_y*dYdLy

def wage_h_y(par, dYdLy):
    return par.mu_y*par.A*par.theta_l_y*dYdLy

def func_Lo(par, sol, t):
    return par.theta_h_o*sol.l_h_o[t] + par.theta_l_o*sol.l_l_o[t]

def func_Ly(par, sol, t):
    return par.theta_h_y*sol.l_h_y[t] + par.theta_l_y*sol.l_l_y[t]


