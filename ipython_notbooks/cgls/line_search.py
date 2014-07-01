import numpy as np

def line_search_newton_raphson(x, d, fd, dfd, iters = 1, tol=1.0e-10):
    """Finds the minimum of the the function f along the direction of d by using a second order Taylor series expansion of f.

    f(x + alpha * d) ~ f(x) + alpha * f'(x) . d + alpha^2 / 2 * dT . f''(x) . d
    therefore alpha = - fd / dfd 
    #
    fd  is a function that evaluates f'(x) . d
    dfd is a function that evaluates dT . f''(x) . d
    #
    returns x2, status
    #
    status is True if dfd is > tol and False otherwise.
    if status is False then the local curvature is negative and the 
    minimum along the line is infinitely far away.
    Algorithm from Eq. (57) of painless-conjugate-gradient.pdf
    """
    for i in range(iters):
        fd_i  = fd(x, d)
        dfd_i = dfd(x, d)
        if dfd_i < tol :
            return x, False
        #
        alpha = - fd_i / dfd_i 
        #
        x = x + alpha * d
    return x, True

def line_search_secant(x, d, fd, iters = 1, sigma = 1.0e-3, tol=1.0e-10):
    """Finds the minimum of the the function f along the direction of d by using a second order Taylor series expansion of f and approximating the second derivative of f with two single derivatives.

    f(x + alpha * d) ~ f(x) + alpha * f'(x) . d + alpha^2 / sigma * [f'(x + sigma * d) . d - f'(x) . d]
    therefore alpha = - sigma * f'(x) . d / [f'(x + sigma * d) . d - f'(x) . d]
    #
    fd  is a function that evaluates f'(x) . d
    #
    returns x2, status
    #
    status is True if [f'(x + sigma * d) . d - f'(x) . d] is > tol and False otherwise.
    if status is False then the local curvature is negative and the 
    minimum along the line is infinitely far away.
    Algorithm from Eq. (59) of painless-conjugate-gradient.pdf
    """
    for i in range(iters):
        fd_0  = fd(x, d)
        fd_1  = fd(x + sigma * d, d)
        dfd   = fd_1 - fd_0
        if dfd < tol :
            return x, False
        #
        alpha = - sigma * fd_0 / dfd 
        #
        x = x + alpha * d
        sigma = - alpha # ??????
    return x, True
