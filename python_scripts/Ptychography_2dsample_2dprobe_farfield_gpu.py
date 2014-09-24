import numpy as np
import scipy as sp
from scipy import ndimage
import os, sys, getopt
from ctypes import *
import bagOfns as bg
import time

class Ptychography_gpu(object):
    def __init__(self, diffs, coords, mask, probe, sample, sample_support, pmod_int = False): 
        """Initialise the Ptychography module with the data in 'inputDir' 
        
        Naming convention:
        coords_100x2.raw            list of y, x coordinates in np.float64 pixel units
        diffs_322x256x512.raw       322 (256,512) diffraction patterns in np.float64
                                    The zero pixel must be at [0, 0] and there must 
                                    be an equal no. of postive and negative frequencies
        mask_256x512.raw            (optional) mask for the diffraction data np.float64
        probeInit_256x512           (optional) Initial estimate for the probe np.complex128
        sampleInit_1024x2048        (optional) initial estimate for the sample np.complex128
                                    also sets the field of view
                                    If not present then initialise with random numbers        
        """
        #
        # Get the shape
        shape  = diffs[0].shape
        #
        # Store these values
        self.exits      = makeExits(sample, probe, coords)
        #
        # This will save time later
        self.diffAmps   = bg.quadshift(np.sqrt(diffs))
        self.shape      = shape
        self.shape_sample = sample.shape
        self.coords     = coords
        self.mask       = bg.quadshift(mask)
        self.probe      = probe
        self.sample     = sample
        self.alpha_div  = 1.0e-10
        self.error_mod  = []
        self.error_sup  = []
        self.error_conv = []
        self.probe_sum  = None
        self.sample_sum = None
        self.diffNorm   = np.sum(self.mask * (self.diffAmps)**2)
        self.pmod_int   = pmod_int
        self.sample_support = sample_support
        #
        # create a gpu thread
        api               = cluda.cuda_api()
        self.thr          = api.Thread.create()
        #
        # send the diffraction amplitudes, the exit waves and the mask to the gpu
        self.diffAmps_gpu = self.thr.to_device(self.diffAmps) * np.sqrt(float(self.diffAmps.shape[1]) * float(self.diffAmps.shape[2]))
        self.exits_gpu    = self.thr.to_device(self.exits)
        mask2             = np.zeros_like(diffs, dtype=np.complex128)
        mask2[:]          = self.mask.astype(np.complex128)
        self.mask_gpu     = self.thr.to_device(mask2)
        #
        # compile the fft routine
        fft               = FFT(self.diffAmps_gpu.astype(np.complex128), axes=(1,2))
        self.fftc         = fft.compile(self.thr, fast_math=True)

    def Thibault_sample(self, iters=1):
        exits2_gpu = self.thr.empty_like(self.exits_gpu)
        print 'i \t\t eConv \t\t eSup'
        for i in range(iters):
            exits = self.exits_gpu.get()
            self.Psup_sample(exits)
            #
            self.thr.to_device(makeExits2(self.sample, self.probe, self.coords, exits), dest=exits2_gpu)
            #
            self.error_sup.append(gpuarray.sum(abs(self.exits_gpu - exits2_gpu)**2).get()/self.diffNorm)
            #
            exits2_gpu = self.exits_gpu + self.Pmod(2*exits2_gpu - self.exits_gpu) - exits2_gpu
            #
            self.error_conv.append(gpuarray.sum(abs(self.exits_gpu - exits2_gpu)**2).get()/self.diffNorm)
            #
            self.error_mod.append(None)
            #
            self.exits_gpu = exits2_gpu.copy()
            #
            update_progress(i / max(1.0, float(iters-1)), 'Thibault sample', i, self.error_conv[-1], self.error_sup[-1])
            #

    def Thibault_probe(self, iters=1):
        exits2_gpu = self.thr.empty_like(self.exits_gpu)
        print 'i \t\t eConv \t\t eSup'
        for i in range(iters):
            exits = self.exits_gpu.get()
            self.Psup_probe(exits)
            #
            self.thr.to_device(makeExits2(self.sample, self.probe, self.coords, exits), dest=exits2_gpu)
            #
            self.error_sup.append(gpuarray.sum(abs(self.exits_gpu - exits2_gpu)**2).get()/self.diffNorm)
            #
            exits2_gpu = self.exits_gpu + self.Pmod(2*exits2_gpu - self.exits_gpu) - exits2_gpu
            #
            self.error_conv.append(gpuarray.sum(abs(self.exits_gpu - exits2_gpu)**2).get()/self.diffNorm)
            #
            self.error_mod.append(None)
            #
            self.exits_gpu = exits2_gpu.copy()
            #
            update_progress(i / max(1.0, float(iters-1)), 'Thibault probe', i, self.error_conv[-1], self.error_sup[-1])
            #

    def Thibault_both(self, iters=1):
        exits2_gpu = self.thr.empty_like(self.exits_gpu)
        print 'i \t\t eMod \t\t eSup'
        for i in range(iters):
            exits = self.exits_gpu.get()
            self.Psup_sample(exits, thresh=1.0)
            self.Psup_probe(exits)
            #
            self.thr.to_device(makeExits2(self.sample, self.probe, self.coords, exits), dest=exits2_gpu)
            #
            self.error_sup.append(gpuarray.sum(abs(self.exits_gpu - exits2_gpu)**2).get()/self.diffNorm)
            #
            exits2_gpu = self.exits_gpu + self.Pmod(2*exits2_gpu - self.exits_gpu) - exits2_gpu
            #
            self.error_conv.append(gpuarray.sum(abs(self.exits_gpu - exits2_gpu)**2).get()/self.diffNorm)
            #
            self.error_mod.append(None)
            #
            self.exits_gpu = exits2_gpu.copy()
            #
            update_progress(i / max(1.0, float(iters-1)), 'Thibault sample', i, self.error_conv[-1], self.error_sup[-1])

    def ERA_sample(self, iters=1):
        exits2_gpu = self.thr.empty_like(self.exits_gpu)
        print 'i, eMod, eSup'
        for i in range(iters):
            exits2_gpu = self.Pmod(self.exits_gpu)
            #
            self.error_mod.append(gpuarray.sum(abs(self.exits_gpu - exits2_gpu)**2).get()/self.diffNorm)
            #
            exits = exits2_gpu.get()
            self.Psup_sample(exits)
            #
            self.thr.to_device(makeExits2(self.sample, self.probe, self.coords, exits), dest=self.exits_gpu)
            #
            self.error_sup.append(gpuarray.sum(abs(self.exits_gpu - exits2_gpu)**2).get()/self.diffNorm)
            #
            update_progress(i / max(1.0, float(iters-1)), 'ERA sample', i, self.error_mod[-1], self.error_sup[-1])

    def ERA_probe(self, iters=1):
        exits2_gpu = self.thr.empty_like(self.exits_gpu)
        print 'i, eMod, eSup'
        for i in range(iters):
            exits2_gpu = self.Pmod(self.exits_gpu)
            #
            self.error_mod.append(gpuarray.sum(abs(self.exits_gpu - exits2_gpu)**2).get()/self.diffNorm)
            #
            exits = exits2_gpu.get()
            self.Psup_probe(exits)
            #
            self.thr.to_device(makeExits2(self.sample, self.probe, self.coords, exits), dest=self.exits_gpu)
            #
            self.error_sup.append(gpuarray.sum(abs(self.exits_gpu - exits2_gpu)**2).get()/self.diffNorm)
            #
            update_progress(i / max(1.0, float(iters-1)), 'ERA probe', i, self.error_mod[-1], self.error_sup[-1])
        
    def ERA_both(self, iters=1):
        exits2_gpu = self.thr.empty_like(self.exits_gpu)
        print 'i, eMod, eSup'
        for i in range(iters):
            exits2_gpu = self.Pmod(self.exits_gpu)
            #
            self.error_mod.append(gpuarray.sum(abs(self.exits_gpu - exits2_gpu)**2).get()/self.diffNorm)
            #
            exits = exits2_gpu.get()
            for j in range(5):
                self.Psup_sample(exits, thresh=1.0)
                self.Psup_probe(exits)
            #
            self.thr.to_device(makeExits2(self.sample, self.probe, self.coords, exits), dest=self.exits_gpu)
            #
            self.error_sup.append(gpuarray.sum(abs(self.exits_gpu - exits2_gpu)**2).get()/self.diffNorm)
            #
            update_progress(i / max(1.0, float(iters-1)), 'ERA both', i, self.error_mod[-1], self.error_sup[-1])

    def Pmod(self, exits_gpu):
        exits2_gpu = self.thr.empty_like(exits_gpu)
        self.fftc(exits2_gpu, exits_gpu)
        #
        exits2_gpu    = exits2_gpu * (self.mask_gpu * self.diffAmps_gpu / (abs(exits2_gpu) + self.alpha_div) + (1.0 - self.mask_gpu))
        #
        self.fftc(exits2_gpu, exits2_gpu, True)
        return exits2_gpu

    def Psup_sample(self, exits, thresh=False, inPlace=True):
        """ """
        sample_out  = np.zeros_like(self.sample)
        # 
        # Calculate denominator
        # but only do this if it hasn't been done already
        # (we must set self.probe_sum = None when the probe/coords has changed)
        if self.probe_sum is None :
            probe_sum   = np.zeros_like(self.sample, dtype=np.float64)
            probe_s     = np.real(np.conj(self.probe) * self.probe)
            for coord in self.coords:
                probe_sum[-coord[0]:self.shape[0]-coord[0], -coord[1]:self.shape[1]-coord[1]] += probe_s
            self.probe_sum = probe_sum
        # 
        # Calculate numerator
        exits     = np.conj(self.probe) * exits
        for coord, exit in zip(self.coords, exits):
            sample_out[-coord[0]:self.shape[0]-coord[0], -coord[1]:self.shape[1]-coord[1]] += exit
        #
        # divide
        sample_out  = sample_out / (self.probe_sum + self.alpha_div)
        sample_out  = sample_out * self.sample_support + ~self.sample_support
        #
        if thresh :
            sample_out = bg.threshold(sample_out, thresh=thresh)
        #
        self.sample_sum = None
        if inPlace :
            self.sample = sample_out
        else :
            return sample_out

    def Psup_probe(self, exits, inPlace=True):
        """ """
        probe_out  = np.zeros_like(self.probe)
        # 
        # Calculate denominator
        # but only do this if it hasn't been done already
        # (we must set self.probe_sum = None when the probe/coords has changed)
        if self.sample_sum is None :
            sample_sum  = np.zeros_like(self.probe, dtype=np.float64)
            temp        = np.real(self.sample * np.conj(self.sample))
            for coord in self.coords:
                sample_sum  += temp[-coord[0]:self.shape[0]-coord[0], -coord[1]:self.shape[1]-coord[1]]
            self.sample_sum = sample_sum + self.alpha_div
        # 
        # Calculate numerator
        # probe = sample * [sum np.conj(sample_shifted) * exit_shifted] / sum |sample_shifted|^2 
        sample_conj = np.conj(self.sample)
        for exit, coord in zip(exits, self.coords):
            probe_out += exit * sample_conj[-coord[0]:self.shape[0]-coord[0], -coord[1]:self.shape[1]-coord[1]]
        #
        # divide
        probe_out   = probe_out / self.sample_sum 
        #
        self.probe_sum = None
        if inPlace:
            self.probe = probe_out
        else : 
            return probe_out


# Example usage
if __name__ == '__main__' :
    pass
