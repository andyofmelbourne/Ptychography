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


def line_search_ERA(x, d, fd, iters = 1, tol=1.0e-2):
    for i in range(iters):
        fd_0 = fd(x, d)
        x    = x - fd_0 * d / np.sqrt(np.sum(np.abs(d)))
        if np.abs(fd_0) < tol :
            return x, True
    return x, True

def line_search_linear(x, d, fd, f, iters = 1, tol=1.0e-1, silent = True):  
    """f(x + alpha * d) ~ f(x) + alpha * f'(x) . d

    alpha = - f(x) / [ f'(x) . d ] 
    """
    a      = 2.0
    f_0    = f(x)
    fd_0   = fd(x, d)
    if silent == False :
        print 'Input error :', f_0
    for i in range(iters):
        #
        # If the slope is too shallow then don't step
        if np.abs(fd_0) < tol :
            if silent == False :
                print 'too shallow'
            return x, f_0, False
        #
        # Update the current position
        alpha  = - f_0 / fd_0 
        x2     = x + a * alpha * d
        f_2    = f(x2)
        #
        # If the error has increased then try halving the step size
        if f_2 < f_0 :
            x     = x2.copy()
            f_0   = f_2
            fd_0  = fd(x, d)
        else :
            if silent == False :
                print 'larger f'
            a = a / 2.0
    if silent == False :
        print 'output error :', f_0
    return x, f_0, True
