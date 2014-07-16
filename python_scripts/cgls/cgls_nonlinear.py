import numpy as np
import line_search as ls


class Steepest(object):
    """Solve a nonlinear optimisation problem using the steepest descent algorithm"""

    def __init__(self, x0, f, df, fd, dfd = None, imax = 10**5, e_tol = 1.0e-10):
        self.f     = f
        self.df    = df
        self.fd    = fd
        self.iters = 0
        self.imax  = imax
        self.e_tol = e_tol
        self.errors = []
        self.x     = x0
        if dfd != None :
            self.dfd         = dfd
            self.line_search = lambda x, d: ls.line_search_newton_raphson(x, d, self.fd, self.dfd, iters = 100, tol=1.0e-10)
        else :
            self.dfd         = None
            #self.line_search = lambda x, d: ls.line_search_secant(x, d, self.fd, iters = 1, sigma = 1.0e-3, tol=1.0e-10)
            self.line_search = lambda x, d: ls.line_search_ERA(x, d, self.fd, iters = 10, tol=0.0e-2)

    def sd(self, iterations = None):
        """Iteratively solve the nonlinear equations using the steepest descent algorithm.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        # 
        if iterations == None :
            iterations = self.imax
        #
        for i in range(iterations):
            #
            # go in the direction of steepest descent
            d     = -self.df(self.x)
            #
            # perform a line search of f along d
            x, status = self.line_search(self.x, d)
            #
            # if status is False then the local curvature is a maximum
            # just do a fixed stepsize in this case
            if status == False :
                self.x = self.x + d
            else :
                self.x = x
            #
            # calculate the error
            self.errors.append(self.f(self.x))
            self.iters = self.iters + 1
            if self.iters > self.imax or (self.errors[-1] < self.e_tol):
                break
        #
        return self.x

class Cgls(object):
    """Minimise the function f using the nonlinear cgls algorithm.
    
    """

    def __init__(self, x0, f, df, fd, dfd = None, imax = 10**5, e_tol = 1.0e-10, silent=True):
        self.f     = f
        self.df    = df
        self.fd    = fd
        self.iters = 0
        self.imax  = imax
        self.e_tol = e_tol
        self.errors = []
        self.x     = x0
        self.silent = silent
        if dfd != None :
            self.dfd         = dfd
            self.line_search = lambda x, d: ls.line_search_newton_raphson(x, d, self.fd, self.dfd, iters = 100, tol=1.0e-10)
        else :
            self.dfd         = None
            self.line_search = lambda x, d: ls.line_search_secant(x, d, self.fd, iters = 5, sigma = 1.0e-2, tol=1.0e-10)
            #self.line_search = lambda x, d: ls.line_search_ERA(x, d, self.fd, iters = 10, tol=0.0e-2)
        #
        #self.cgls = self.cgls_Ploak_Ribiere
        self.cgls = self.cgls_Flecher_Reeves

    def cgls_Flecher_Reeves(self, iterations = None):
        """
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iters == 0 :
            self.r         = - self.df(self.x)
            self.d         = self.r.copy()
            self.delta_new = np.sum(self.r**2)
            self.delta_0   = self.delta_new.copy()
        # 
        if iterations == None :
            iterations = self.imax
        #
        for i in range(iterations):
            #
            # perform a line search of f along d
            # self.x, status = self.line_search(self.x, self.d)
            #
            # If this fails then try a backup line-search
            #if status == False :
            #    print 'fallng back to line_search_linear...'
            #    self.x, status = ls.line_search_linear(self.x, self.d, self.fd, self.f, iters=5)
            self.x, error, status = ls.line_search_linear(self.x, self.d, self.fd, self.f, iters=10, silent = self.silent)
            # 
            self.r         = - self.df(self.x)
            delta_old      = self.delta_new
            self.delta_new = np.sum(self.r**2)
            #
            # Fletcher-Reeves formula
            beta           = self.delta_new / delta_old
            #
            self.d         = self.r + beta * self.d
            #
            # reset the algorithm 
            if (self.iters % 4 == 0) or (status == False) :
                if self.silent == False :
                    print 'reseting ...'
                self.d = self.r.copy()
            #
            # calculate the error
            self.errors.append(error)
            self.iters = self.iters + 1
            if self.iters > self.imax or (self.errors[-1] < self.e_tol):
                break
        #
        #
        return self.x

    def cgls_Ploak_Ribiere(self, iterations = None):
        """
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iters == 0 :
            self.r         = - self.df(self.x)
            self.r_old     = self.r.copy()
            self.d         = self.r.copy()
            self.delta_new = np.sum(self.r**2)
            self.delta_0   = self.delta_new.copy()
        # 
        if iterations == None :
            iterations = self.imax
        #
        for i in range(iterations):
            #
            # perform a line search of f along d
            self.x, status = self.line_search(self.x, self.d)
            # 
            self.r         = - self.df(self.x)
            delta_old      = self.delta_new
            delta_mid      = np.sum(self.r * self.r_old)
            self.r_old     = self.r.copy()
            self.delta_new = np.sum(self.r**2)
            #
            # Polak-Ribiere formula
            beta           = (self.delta_new - delta_mid)/ delta_old
            #
            self.d         = self.r + beta * self.d
            #
            # reset the algorithm 
            if (self.iters % 50 == 0) or (status == False) or beta <= 0.0 :
                self.d = self.r
            else :
                self.d = self.r + beta * self.d
            #
            # calculate the error
            self.errors.append(self.f(self.x))
            self.iters = self.iters + 1
            if self.iters > self.imax or (self.errors[-1] < self.e_tol):
                break
        #
        #
        return self.x
