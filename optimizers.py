# -*- coding: utf-8 -*-
"""golden section search

Numba JIT compilled golden section search optimizer for a custom objective.
"""

import math
import numpy as np
from numba import njit
from math import copysign, isfinite

from jit_module import jit_if_enabled

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

@jit_if_enabled()
def golden_section_int_modified(a, b, f, *args):
    """
    Integer-valued golden-section-like search on [a, b].

    Rule:
    1. Prefer positive values.
    2. Among positive values, choose the one closest to zero.
    3. If no positive values exist, choose the non-positive value closest to zero.
    """
    if f is None:
        raise ValueError("Provide a callable f(x, *args).")

    if a > b:
        a, b = b, a

    phi = (np.sqrt(5) - 1) / 2
    cache = {}

    best_pos_x = None
    best_pos_f = np.inf

    best_nonpos_x = None
    best_nonpos_f = -np.inf

    def register(x, fx):
        nonlocal best_pos_x, best_pos_f, best_nonpos_x, best_nonpos_f

        if np.isnan(fx):
            raise ValueError(f"f({x}, *args) returned NaN.")

        if fx > 0 and fx < best_pos_f:
            best_pos_x = x
            best_pos_f = fx

        if fx <= 0 and fx > best_nonpos_f:
            best_nonpos_x = x
            best_nonpos_f = fx

    def eval_f(x):
        x = int(x)
        if x not in cache:
            cache[x] = f(x, *args)
        return cache[x]

    while b - a > 2:
        c = a + int(np.floor((1 - phi) * (b - a)))
        d = a + int(np.floor(phi * (b - a)))

        if c == d:
            break

        fa = eval_f(a)
        fb = eval_f(b)

        register(a, fa)
        register(b, fb)

        if np.sign(fa) == np.sign(fb):
            if fa == 0 and fb == 0:
                break

            if fa < 0:
                if fb > fa:
                    a = c
                else:
                    b = d

            elif fa > 0:
                if fb < fa:
                    a = c
                else:
                    b = d

        else:
            fc = eval_f(c)
            fd = eval_f(d)

            register(c, fc)
            register(d, fd)

            if np.sign(fc) > 0 and np.sign(fd) > 0:
                if fc < fd:
                    b = c
                else:
                    a = d
            else:
                if np.sign(fc) > 0 and np.sign(fd) < 0:
                    a = c
                else:
                    b = d

    for x in range(int(a), int(b) + 1):
        fx = eval_f(x)
        register(x, fx)

    if best_pos_x is not None:
        return best_pos_x, best_pos_f

    if best_nonpos_x is not None:
        return best_nonpos_x, best_nonpos_f

    raise ValueError("No valid points found in the interval.")





@jit_if_enabled()
def golden_section_modified(a, b, f, *args, tol=1e-8, max_iter=1_000):
    """
    Continuous golden-section-like search on [a, b].

    Rule:
    1. Prefer positive values.
    2. Among positive values, choose the one closest to zero.
    3. If no positive values exist, choose the non-positive value closest to zero.
    """
    if f is None:
        raise ValueError("Provide a callable f(x, *args).")

    if a > b:
        a, b = b, a

    phi = (np.sqrt(5) - 1) / 2

    best_pos_x = np.nan
    best_pos_f = np.inf

    best_nonpos_x = np.nan
    best_nonpos_f = -np.inf

    def register(x, fx):
        nonlocal best_pos_x, best_pos_f, best_nonpos_x, best_nonpos_f

        if np.isnan(fx):
            raise ValueError("f(x, *args) returned NaN.")

        if fx > 0.0 and fx < best_pos_f:
            best_pos_x = x
            best_pos_f = fx

        if fx <= 0.0 and fx > best_nonpos_f:
            best_nonpos_x = x
            best_nonpos_f = fx

    for _ in range(max_iter):

        if abs(b - a) <= tol:
            break

        c = a + (1.0 - phi) * (b - a)
        d = a + phi * (b - a)

        fa = f(a, *args)
        fb = f(b, *args)
        fc = f(c, *args)
        fd = f(d, *args)

        register(a, fa)
        register(b, fb)
        register(c, fc)
        register(d, fd)

        if np.sign(fa) != np.sign(fb):
            if np.sign(fc) != np.sign(fa):
                b = c
            elif np.sign(fd) != np.sign(fb):
                a = d
            else:
                a = c
                b = d

        else:
            if fa <= 0.0 and fb <= 0.0:
                if fa > fb:
                    b = d
                else:
                    a = c

            elif fa > 0.0 and fb > 0.0:
                if fa < fb:
                    b = d
                else:
                    a = c

            else:
                a = c
                b = d

    x_mid = 0.5 * (a + b)
    f_mid = f(x_mid, *args)
    register(x_mid, f_mid)

    if not np.isnan(best_pos_x):
        return best_pos_x, best_pos_f

    if not np.isnan(best_nonpos_x):
        return best_nonpos_x, best_nonpos_f

    raise ValueError("No valid points found in the interval.")