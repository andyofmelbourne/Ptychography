import numpy as np
import scipy as sp
from scipy import ndimage
import os, sys, getopt
from ctypes import *
import bagOfns as bg
import time
from Ptychograph_2dsample_2dprobe_farfield import Ptychography

class Ptychography_1dsample(Ptychography):
    def __init__(self, diffs, coords, mask, probe, sample_1d, sample_support_1d, pmod_int = False): 
        from cgls import cgls_nonlinear
        self.sample_1d = sample_1d
        self.sample_support_1d = sample_support_1d
        #
        sample         = np.zeros((probe.shape[0], sample_1d.shape[0]), dtype = sample_1d.dtype)
        sample_support = np.zeros((probe.shape[0], sample_1d.shape[0]), dtype = sample_1d.dtype)
        sample[:]         = sample_1d.copy()
        sample_support[:] = sample_support_1d.copy()
        #
        Ptychography.__init__(self, diffs, coords, mask, probe, sample, sample_support, pmod_int)
        #
        # nonlinear cgls update for coordinates in 1d
        n  = len(self.coords) - 1
        f  = lambda x   : self.f(x, n = n)
        fd = lambda x, d: self.grad_f_dot_d(x, d, n = n)
        df = lambda x   : self.grad_f(x, n = n)
        self.cg = cgls_nonlinear.Cgls(self.coords_contract(self.coords, n = n), f, df, fd)

    def Psup_sample(self, exits, thresh=False, inPlace=True):
        """ """
        sample_out  = np.zeros_like(self.sample)
        # 
        # Calculate denominator
        # but only do this if it hasn't been done already
        # (we must set self.probe_sum = None when the probe/coords has changed)
        if self.probe_sum is None :
            probe_sum   = np.zeros_like(self.sample, dtype=np.float64)
            probe_large = np.zeros_like(self.sample, dtype=np.float64)
            probe_large[:self.shape[0], :self.shape[1]] = np.real(np.conj(self.probe) * self.probe)
            for coord in self.coords:
                probe_sum  += bg.roll(probe_large, [-coord[0], -coord[1]])
            self.probe_sum = probe_sum.copy()
        # 
        # Calculate numerator
        exits2     = np.conj(self.probe) * exits 
        temp       = np.zeros_like(self.sample)
        for i in range(len(self.coords)):
            temp[:self.shape[0], :self.shape[1]] = exits2[i]
            sample_out += bg.roll(temp, [-self.coords[i][0], -self.coords[i][1]])
        # 
        # project to 1d
        sample_1d = np.sum(sample_out, axis=0)
        #
        # divide
        sample_1d = sample_1d / (np.sum(self.probe_sum, axis=0) + self.alpha_div)
        sample_1d = sample_1d * self.sample_support_1d + ~self.sample_support_1d
        # 
        # expand
        sample_out[:] = sample_1d.copy()
        #
        if thresh :
            sample_out = bg.threshold(sample_out, thresh=thresh)
        if inPlace :
            self.sample    = sample_out
            self.sample_1d = sample_1d
        self.sample_sum = None
        # 
        return sample_out

    #------------------------------------------------------
    # coordinates update stuff
    #------------------------------------------------------
    #
    # we are only solving for a subset of the coordinates
    def coords_expand(self, coords_sub, n = 1):
        coords_full         = self.coords.copy().astype(np.float64)
        coords_full[-n:, 1] = coords_sub
        return coords_full 
    #
    def coords_contract(self, coords_full, n = 1):
        coords_sub         = coords_full[-n:, 1].copy().astype(np.float64)
        return coords_sub 
    #
    def coords_expand_d(self, coords_sub, n = 1):
        coords_full         = np.zeros_like(self.coords, dtype = np.float64)
        coords_full[-n:, 1] = coords_sub
        return coords_full 
    #
    def f(self, x, n = 1):
        x_full = self.coords_expand(x, n = n)
        return fmod(self, x_full)
    #
    def grad_f_dot_d(self, x, d, n = 1):
        x_full = self.coords_expand(x, n = n)
        d_full = self.coords_expand_d(d, n = n)
        return emod_grad_dot_coords_1d(d_full, self.sample, self.probe, x_full, self.diffAmps)
    # 
    def grad_f(self, x, n = 1):
        x_full       = self.coords_expand(x, n = n)
        xx           = emod_grad_coords_1d(self.sample, self.probe, x_full, self.diffAmps)
        x_full[:, 1] = xx    
        return self.coords_contract(x_full, n = n)

    def coords_update_1d(self, iters=1, inPlace=True, intefy=True):
        if intefy :
            self.coords = self.coords.copy().astype(np.float64)
        print 'i \t\t eMod \t\t conv'
        for i in range(iters):
            x = self.cg.cgls(iterations = 1)
            #
            self.error_mod.append(self.cg.errors[-1]/self.diffNorm)
            #
            update_progress(i / max(1.0, float(iters-1)), 'cgls coords', i, self.error_mod[-1], 999)
            # 
            if i >= 1 :
                if (np.abs(self.error_mod[-2] - self.error_mod[-1]) / self.error_mod[-2]) < 1.0e-5 :
                    print 'converged...'
                    break
        #
        coords = self.coords_expand(x, len(self.coords) - 1)
        #
        # If intefy is true then round to the nearest integer
        if intefy :
            coords = np.round(coords).astype(np.int32)
        #
        self.sample_sum = None
        self.probe_sum = None
        if inPlace :
            self.coords = coords
        else :
            return coords


# Example usage
if __name__ == '__main__' :
    pass
