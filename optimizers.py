# -*- coding: utf-8 -*-
"""golden section search

Numba JIT compilled golden section search optimizer for a custom objective.
"""

import math
import numpy as np
from numba import njit
from math import copysign, isfinite

# @njit
def golden(obj,a,b,args=(),tol=1e-6):
    """ golden section search optimizer
    
    Args:

        obj (callable): 1d function to optimize over
        a (double): minimum of starting bracket
        b (double): maximum of starting bracket
        args (tuple): additional arguments to the objective function
        tol (double,optional): tolerance

    Returns:

        (float): optimization result
    
    """
    
    inv_phi = (np.sqrt(5) - 1) / 2 # 1/phi                                                                                                                
    inv_phi_sq = (3 - np.sqrt(5)) / 2 # 1/phi^2     
        
    # a. distance
    dist = b - a
    if dist <= tol: 
        return (a+b)/2

    # b. number of iterations
    n = int(np.ceil(np.log(tol/dist)/np.log(inv_phi)))

    # c. potential new mid-points
    c = a + inv_phi_sq * dist
    d = a + inv_phi * dist
    yc = obj(c,*args)
    yd = obj(d,*args)

    # d. loop
    for _ in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            dist = inv_phi*dist
            c = a + inv_phi_sq * dist
            yc = obj(c,*args)
        else:
            a = c
            c = d
            yc = yd
            dist = inv_phi*dist
            d = a + inv_phi * dist
            yd = obj(d,*args)

    # e. return
    if yc < yd:
        return (a+d)/2
    else:
        return (c+b)/2
    



def brentq(f, a, b, args=(), xtol=1e-12, rtol=4.440892098500626e-16, maxiter=100):
    """
    Brent's root-finding method on [a, b].

    Parameters
    ----------
    f : callable
        Function f(x, *args).
    a, b : float
        Interval endpoints. f(a) and f(b) must have opposite signs.
    args : tuple
        Extra arguments passed to f.
    xtol : float
        Absolute tolerance.
    rtol : float
        Relative tolerance.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    float
        Approximate root.
    """
    fa = f(a, *args)
    fb = f(b, *args)

    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    c = a
    fc = fa
    d = e = b - a

    for _ in range(maxiter):
        if fb == 0:
            return b

        # Make sure that b is the best approximation so far
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        tol = 2.0 * rtol * abs(b) + xtol
        m = 0.5 * (c - b)

        # Converged?
        if abs(m) <= tol:
            return b

        if abs(e) >= tol and abs(fa) > abs(fb):
            # Attempt interpolation
            s = fb / fa
            if a == c:
                # Secant step
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                # Inverse quadratic interpolation
                q_ = fa / fc
                r_ = fb / fc
                p = s * (2.0 * m * q_ * (q_ - r_) - (b - a) * (r_ - 1.0))
                q = (q_ - 1.0) * (r_ - 1.0) * (s - 1.0)

            if p > 0:
                q = -q
            p = abs(p)

            cond1 = 2.0 * p < min(3.0 * abs(m) * abs(q) - abs(tol * q), abs(e * q))
            cond2 = p < abs(0.5 * e * q)

            if cond1 and cond2 and q != 0:
                e = d
                d = p / q
            else:
                d = m
                e = m
        else:
            d = m
            e = m

        a = b
        fa = fb

        if abs(d) > tol:
            b += d
        else:
            b += tol if m > 0 else -tol

        fb = f(b, *args)

        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c = a
            fc = fa
            d = e = b - a

    raise RuntimeError("Maximum iterations exceeded")