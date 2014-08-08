# This code is primarily designed to run from the command 
# line with a configuration file as input from the user.
import numpy as np
import scipy as sp
from scipy import ndimage
import os, sys, getopt
from ctypes import *
import bagOfns as bg
import time
#
# GPU stuff 
try :
    import pycuda.autoinit 
    import pycuda.gpuarray as gpuarray
    import pycuda.cumath as cumath
    from reikna.fft import FFT
    import reikna.cluda as cluda
    GPU_calc = True
except :
    GPU_calc = False

GPU_calc = False

print 'GPU_calc', GPU_calc

def update_progress(progress, algorithm, i, emod, esup):
    barLength = 15 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\r{0}: [{1}] {2}% {3} {4} {5} {6} {7}".format(algorithm, "#"*block + "-"*(barLength-block), int(progress*100), i, emod, esup, status, " " * 5) # this last bit clears the line
    sys.stdout.write(text)
    sys.stdout.flush()

def main(argv):
    inpurtdir = './'
    outputdir = './'
    try :
        opts, args = getopt.getopt(argv,"hi:o:",["inputdir=","outputdir="])
    except getopt.GetoptError:
        print 'process_diffs.py -i <inputdir> -o <outputdir>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'process_diffs.py -i <inputdir> -o <outputdir>'
            sys.exit()
        elif opt in ("-i", "--inputdir"):
            inputdir = arg
        elif opt in ("-o", "--outputdir"):
            outputdir = arg
    return inputdir, outputdir

def fnamBase_match(fnam):
    fnam_base  = os.path.basename(fnam)
    fnam_dir   = os.path.abspath(os.path.dirname(fnam))
    onlyfiles  = [ f for f in os.listdir(fnam_dir) if os.path.isfile(os.path.join(fnam_dir,f)) ]
    fnam_match = [ f for f in onlyfiles if f[:len(fnam_base)] == fnam_base ]
    try : 
        fnam_match[0]
    except :
        return False
    return True


class Ptychography(object):
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

    def ERA_sample(self, iters=1):
        print 'i, eMod, eSup'
        for i in range(iters):
            exits = self.Pmod(self.exits)
            self.exits -= exits
            self.error_mod.append(np.sum(np.real(np.conj(self.exits)*self.exits))/self.diffNorm)
            #
            self.Psup_sample(exits)
            self.exits = makeExits(self.sample, self.probe, self.coords)
            exits     -= self.exits
            self.error_sup.append(np.sum(np.real(np.conj(exits)*exits))/self.diffNorm)
            #
            update_progress(i / max(1.0, float(iters-1)), 'ERA sample', i, self.error_mod[-1], self.error_sup[-1])

    def ERA_probe(self, iters=1):
        print 'i \t\t eMod \t\t eSup'
        for i in range(iters):
            exits = self.Pmod(self.exits)
            self.exits -= exits
            self.error_mod.append(np.sum(np.real(np.conj(self.exits)*self.exits))/self.diffNorm)
            #
            self.Psup_probe(exits)
            self.exits = makeExits(self.sample, self.probe, self.coords)
            exits     -= self.exits
            self.error_sup.append(np.sum(np.real(np.conj(exits)*exits))/self.diffNorm)
            #
            update_progress(i / max(1.0, float(iters-1)), 'ERA Probe', i, self.error_mod[-1], self.error_sup[-1])

    def ERA_both(self, iters=1):
        print 'i, eMod, eSup'
        for i in range(iters):
            exits = self.Pmod(self.exits)
            #
            self.exits -= exits
            self.error_mod.append(np.sum(np.real(np.conj(self.exits)*self.exits))/self.diffNorm)
            #
            for j in range(5):
                self.Psup_sample(exits, thresh=1.0)
                self.Psup_probe(exits)
            #
            self.exits = makeExits(self.sample, self.probe, self.coords)
            exits     -= self.exits
            self.error_sup.append(np.sum(np.real(np.conj(exits)*exits))/self.diffNorm)
            #
            update_progress(i / max(1.0, float(iters-1)), 'ERA both', i, self.error_mod[-1], self.error_sup[-1])

    def HIO_sample(self, iters=1, beta=1):
        print 'i \t\t eMod \t\t eSup'
        for i in range(iters):
            exits = self.Pmod(self.exits)
            self.Psup_sample((1 + 1/beta)*exits - 1/beta * self.exits)
            exits = self.exits + beta * makeExits(self.sample, self.probe, self.coords) - beta * exits
            #
            self.exits = self.exits - self.Pmod(exits)
            self.error_mod.append(np.sum(np.real(np.conj(self.exits)*self.exits))/self.diffNorm)
            #
            self.Psup_sample(exits)
            self.exits = exits - makeExits(self.sample, self.probe, self.coords)
            self.error_sup.append(np.sum(np.real(np.conj(self.exits)*self.exits))/self.diffNorm)
            #
            update_progress(i / max(1.0, float(iters-1)), 'HIO sample', i, self.error_mod[-1], self.error_sup[-1])
            #
            self.exits = exits

    def HIO_probe(self, iters=1, beta=1):
        print 'i \t\t eMod \t\t eSup'
        for i in range(iters):
            exits = self.Pmod(self.exits)
            self.Psup_probe((1 + 1/beta)*exits - 1/beta * self.exits)
            exits = self.exits + beta * makeExits(self.sample, self.probe, self.coords) - beta * exits
            #
            self.exits = exits - self.Pmod(exits)
            self.error_mod.append(np.sum(np.real(np.conj(self.exits)*self.exits))/self.diffNorm)
            #
            self.Psup_probe(exits)
            self.exits = exits - makeExits(self.sample, self.probe, self.coords)
            self.error_sup.append(np.sum(np.real(np.conj(self.exits)*self.exits))/self.diffNorm)
            #
            update_progress(i / max(1.0, float(iters-1)), 'HIO probe', i, self.error_mod[-1], self.error_sup[-1])
            #
            self.exits = exits

    def Thibault_sample(self, iters=1):
        print 'i \t\t eConv \t\t eSup'
        for i in range(iters):
            self.Psup_sample(self.exits, thresh=1.0)
            exits = makeExits(self.sample, self.probe, self.coords)
            self.error_sup.append(np.sum(np.abs(exits - self.exits)**2)/self.diffNorm)
            exits = self.exits + self.Pmod(2*exits - self.exits) - exits
            self.error_conv.append(np.sum(np.abs(exits - self.exits)**2)/self.diffNorm)
            self.error_mod.append(None)
            self.exits = exits
            #
            update_progress(i / max(1.0, float(iters-1)), 'Thibault sample', i, self.error_conv[-1], self.error_sup[-1])

    def Thibault_probe(self, iters=1):
        print 'i \t\t eConv \t\t eSup'
        for i in range(iters):
            self.Psup_probe(self.exits)

            exits = makeExits(self.sample, self.probe, self.coords)

            self.error_sup.append(np.sum(np.abs(exits - self.exits)**2)/self.diffNorm)

            exits = self.exits + self.Pmod(2*exits - self.exits) - exits

            self.error_conv.append(np.sum(np.abs(exits - self.exits)**2)/self.diffNorm)
            self.error_mod.append(None)

            self.exits = exits
            #
            update_progress(i / max(1.0, float(iters-1)), 'Thibault probe', i, self.error_conv[-1], self.error_sup[-1])

    def Thibault_both(self, iters=1):
        print 'i \t\t eMod \t\t eSup'
        for i in range(iters):
            self.Psup_sample(self.exits, thresh=1.0)
            self.Psup_probe(self.exits)
            self.Psup_sample(self.exits, thresh=1.0)
            self.Psup_probe(self.exits)

            exits = makeExits(self.sample, self.probe, self.coords)

            self.error_sup.append(np.sum(np.abs(exits - self.exits)**2)/self.diffNorm)

            exits = self.exits + self.Pmod(2*exits - self.exits) - exits

            self.error_conv.append(np.sum(np.abs(exits - self.exits)**2)/self.diffNorm)
            self.error_mod.append(None)

            self.exits = exits
            #
            update_progress(i / max(1.0, float(iters-1)), 'Thibault sample / probe', i, self.error_conv[-1], self.error_sup[-1])

    def Huang(self, iters=None):
        """This is the algorithm used in "11 nm Hard X-ray focus from a large-aperture multilayer Laue lens" (2013) nature"""
        def randsample():
            array = np.random.random(self.sample.shape) + 1J * np.random.random(self.sample.shape) 
            return array
        #
        for i in range(5):
            self.sample     = randsample()
            self.sample_sum = None
            self.exits = makeExits(self.sample, self.probe, self.coords)
            #
            self.Thibault_sample(iters=5)
            #
            self.Thibault_both(iters=75)
            #
            probes = []
            for j in range(20):
                self.Thibault_both(iters=1)
                probes.append(self.probe.copy())
            self.probe     = np.sum(np.array(probes), axis=0) / float(len(probes))
            self.probe_sum = None
        #
        probe  = self.probe.copy()
        probes  = []
        samples = []
        for i in range(10):
            self.sample     = randsample()
            self.sample_sum = None
            self.probe      = probe.copy()
            self.probe_sum  = None
            self.exits = makeExits(self.sample, self.probe, self.coords)
            #
            self.Thibault_sample(iters=5)
            #
            self.Thibault_both(iters=75)
            #
            for j in range(20):
                self.Thibault_both(iters=1)
                probes.append(self.probe.copy())
                samples.append(self.sample.copy())
        self.probe = np.sum(np.array(probes), axis=0) / float(len(probes))
        self.sample = np.sum(np.array(samples), axis=0) / float(len(samples))
        
    def Pmod(self, exits, pmod_int = False):
        exits_out = bg.fftn(exits)
        if self.pmod_int or pmod_int :
            exits_out = exits_out * (self.mask * self.diffAmps * (self.diffAmps > 0.99) /  np.clip(np.abs(exits_out), self.alpha_div, np.inf) \
                           + (~self.mask) )
        else :
            exits_out = exits_out * (self.mask * self.diffAmps / np.clip(np.abs(exits_out), self.alpha_div, np.inf) \
                           + (~self.mask) )
            #exits_out = exits_out * self.diffAmps / (np.abs(exits_out) + self.alpha_div)
        exits_out = bg.ifftn(exits_out)
        return exits_out

    def Pmod_hat(self, exits_F, pmod_int = False):
        exits_out = exits_F
        if self.pmod_int or pmod_int :
            exits_out = exits_out * (self.mask * self.diffAmps * (self.diffAmps > 0.99) /  np.clip(np.abs(exits_out), self.alpha_div, np.inf) \
                           + (~self.mask) )
        else :
            exits_out = exits_out * (self.mask * self.diffAmps / np.clip(np.abs(exits_out), self.alpha_div, np.inf) \
                           + (~self.mask) )
            #exits_out = exits_out * self.diffAmps / (np.abs(exits_out) + self.alpha_div)
        return exits_out

    def Pmod_integer(self, exits):
        """If the diffraction data is in units of no. of particles then there should be no numbers in 
        self.diffAmps in the range (0, 1). This can help us make the modulus substitution more robust.
        """
        exits_out = bg.fftn(exits)
        exits_out = exits_out * (self.mask * self.diffAmps * (self.diffAmps > 0.99) /  np.abs(exits_out) + (~self.mask) )
        exits_out = bg.ifftn(exits_out)
        return exits_out

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
            temp.fill(0.0)
            temp[:self.shape[0], :self.shape[1]] = exits2[i]
            sample_out += bg.roll(temp, [-self.coords[i][0], -self.coords[i][1]])
        #
        # divide
        sample_out  = sample_out / (self.probe_sum + self.alpha_div)
        sample_out  = sample_out * self.sample_support + ~self.sample_support
        #
        if thresh :
            sample_out = bg.threshold(sample_out, thresh=thresh)
        #
        if inPlace :
            self.sample = sample_out
        self.sample_sum = None
        # 
        return sample_out
    
    def Psup_probe(self, exits, inPlace=True):
        """ """
        probe_out  = np.zeros_like(self.probe)
        # 
        # Calculate denominator
        # but only do this if it hasn't been done already
        # (we must set self.probe_sum = None when the probe/coords has changed)
        if self.sample_sum is None :
            sample_sum  = np.zeros_like(self.sample, dtype=np.float64)
            temp        = np.real(self.sample * np.conj(self.sample))
            for coord in self.coords:
                sample_sum  += bg.roll(temp, coord)
            self.sample_sum = sample_sum[:self.shape[0], :self.shape[1]]
        # 
        # Calculate numerator
        # probe = sample * [sum np.conj(sample_shifted) * exit_shifted] / sum |sample_shifted|^2 
        for i in range(len(self.coords)):
            sample_shifted = bg.roll(self.sample, self.coords[i])[:self.shape[0], :self.shape[1]]
            probe_out     += np.conj(sample_shifted) * exits[i] 
        #
        # divide
        probe_out   = probe_out / (self.sample_sum + self.alpha_div)
        if inPlace:
            self.probe = probe_out
        self.probe_sum = None
        # 
        return probe_out

    def back_prop(self, iters=1):
        """Back propagate from the diffraction patterns using the phase from probe in the detector plane.

        Returns an array of exit waves:
        exit i = ifft( sqrt(diff i) * e^(i phase_of_probe_i_in diffraction_plane) )
        Fill the masked area with the amplitude from the probe.
        iters is a dummy argument (for consistency with ERA_sample and such)
        """
        probeF    = bg.fft2(self.probe, origin='zero')
        exits_out = np.zeros_like(self.exits)
        exits_out = probeF * (self.mask * self.diffAmps / (self.alpha_div + np.abs(probeF)) \
                       + (~self.mask) )
        self.exits = bg.ifftn(exits_out)
    


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
        self.probeInit  = probe.copy()
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
            update_progress(i / max(1.0, float(iters-1)), 'Thibault both', i, self.error_conv[-1], self.error_sup[-1])

    def Thibault_both_av(self, iters=1):
        """Average the output sample and probe from each iteration to reduce high frequency artefacts."""
        exits2_gpu = self.thr.empty_like(self.exits_gpu)
        samples = []
        probes = []
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
            update_progress(i / max(1.0, float(iters-1)), 'Thibault both averaging', i, self.error_conv[-1], self.error_sup[-1])
            #
            samples.append(self.sample.copy())
            probes.append(self.probe.copy())
        #
        self.sample     = np.sum(np.array(samples), axis=0) / float(iters)
        self.probe      = np.sum(np.array(probes), axis=0) / float(iters)
        self.sample_sum = None
        self.probe_sum  = None

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
            for j in range(1):
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

    def Pmod_probe(self, iters = 1, inPlace=True, mask = False):
        """ """
        print 'applying the modulus constraint to the probe...'
        probe_out  = bg.fftn(self.probe)
        if mask :
            probe_out  = probe_out * (self.mask * np.abs(bg.fftn(self.probeInit)) / np.clip(np.abs(probe_out), self.alpha_div, np.inf) + (~self.mask) )
        else :
            probe_out  = probe_out * np.abs(bg.fftn(self.probeInit)) / np.clip(np.abs(probe_out), self.alpha_div, np.inf) 
        probe_out  = bg.ifftn(probe_out)
        #
        self.probe_sum = None
        if inPlace:
            self.probe = probe_out
        else : 
            return probe_out

    def randsample(self):
        array = np.random.random(self.sample.shape) + 1J * np.random.random(self.sample.shape) 
        return array

    def Huang(self, iters=None):
        """This is the algorithm used in "11 nm Hard X-ray focus from a large-aperture multilayer Laue lens" (2013) nature"""
        #
        for i in range(0):
            self.sample     = self.randsample()
            self.sample_sum = None
            self.thr.to_device(makeExits2(self.sample, self.probe, self.coords, self.exits), dest=self.exits_gpu)
            #
            self.Thibault_sample(iters=20)
            self.ERA_sample(30)
            #
            self.Thibault_both(iters=5)
            self.Pmod_probe()
            #
            probes = []
            for j in range(5):
                self.Thibault_both(iters=1)
                self.Pmod_probe()
                probes.append(self.probe.copy())
            self.probe     = np.sum(np.array(probes), axis=0) / float(len(probes))
            self.probe_sum = None
        #
        probe  = self.probe.copy()
        probes  = []
        samples = []
        for i in range(10):
            self.sample     = self.randsample()
            self.sample_sum = None
            self.probe      = probe.copy()
            self.probe_sum  = None
            self.thr.to_device(makeExits2(self.sample, self.probe, self.coords, self.exits), dest=self.exits_gpu)
            #
            self.Thibault_sample(iters=50)
            #
            self.Thibault_both(iters=5)
            self.Pmod_probe()
            #
            self.ERA_both(100)
            #
            for j in range(1):
                #self.Thibault_both(iters=1)
                probes.append(self.probe.copy())
                samples.append(self.sample.copy())
        self.probe = np.sum(np.array(probes), axis=0) / float(len(probes))
        self.sample = np.sum(np.array(samples), axis=0) / float(len(samples))

class Ptychography_1dsample_gpu(Ptychography_gpu):
    def __init__(self, diffs, coords, mask, probe, sample_1d, sample_support_1d, pmod_int = False): 
        from cgls import cgls_nonlinear
        self.sample_1d         = sample_1d
        self.sample_support_1d = sample_support_1d
        #
        sample         = np.zeros((probe.shape[0], sample_1d.shape[0]), dtype = sample_1d.dtype)
        sample_support = np.zeros((probe.shape[0], sample_1d.shape[0]), dtype = sample_1d.dtype)
        sample[:]         = sample_1d.copy()
        sample_support[:] = sample_support_1d.copy()
        #
        Ptychography_gpu.__init__(self, diffs, coords, mask, probe, sample, sample_support, pmod_int)
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
            probe_sum   = np.zeros_like(self.sample_1d, dtype=np.float64)
            probe_s     = np.real(np.conj(self.probe) * self.probe)
            probe_s     = np.sum(probe_s, axis=0)
            for coord in self.coords:
                probe_sum[-coord[1]:self.shape[1]-coord[1]] += probe_s
            self.probe_sum = probe_sum + self.alpha_div
        # 
        # Calculate numerator
        exits     = np.conj(self.probe) * exits
        exits     = np.sum(exits, axis=1)
        sample_1d = np.zeros_like(self.sample_1d)
        for coord, exit in zip(self.coords, exits):
            sample_1d[-coord[1]:self.shape[1]-coord[1]] += exit
        #
        # divide
        sample_1d = sample_1d / self.probe_sum
        sample_1d = sample_1d * self.sample_support_1d + ~self.sample_support_1d
        # 
        # expand
        sample_out[:] = sample_1d
        #
        if thresh :
            sample_out = bg.threshold(sample_out, thresh=thresh)
        self.sample_sum = None
        if inPlace :
            self.sample    = sample_out
            self.sample_1d = sample_1d
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
            sample_sum     = np.zeros_like(self.probe, dtype=np.float64)
            sample_sum_1d  = np.zeros(self.probe.shape[1], dtype=np.float64)
            temp           = np.real(self.sample[0,:] * np.conj(self.sample[0,:]))
            for coord in self.coords:
                sample_sum_1d  += temp[-coord[1]:self.shape[1]-coord[1]]
            sample_sum[:] = sample_sum_1d
            self.sample_sum = sample_sum + self.alpha_div
        # 
        # Calculate numerator
        # probe = sample * [sum np.conj(sample_shifted) * exit_shifted] / sum |sample_shifted|^2 
        sample_conj = np.conj(self.sample[0,:])
        for exit, coord in zip(exits, self.coords):
            probe_out += exit * sample_conj[-coord[1]:self.shape[1]-coord[1]]
        #
        # divide
        probe_out   = probe_out / self.sample_sum 
        #
        self.probe_sum = None
        if inPlace:
            self.probe = probe_out
        else : 
            return probe_out

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
                if (np.abs(self.error_mod[-2] - self.error_mod[-1]) / self.error_mod[-2]) < 1.0e-8 :
                    print 'converged...'
                    break
        #
        coords = self.coords_expand(x, len(self.coords) - 1)
        print ''
        print 'delta coordates:'
        print coords - self.coords
        #
        # If intefy is true then round to the nearest integer
        if intefy :
            coords = np.round(coords).astype(np.int32)
        #
        # Let's reset the sample and start fresh
        self.sample    = np.ones_like(self.sample)
        self.sample_1d = np.ones_like(self.sample_1d)
        self.sample_sum = None
        self.probe_sum = None
        if inPlace :
            self.coords = coords
        else :
            return coords


#------------------------------------------------------
# coordinates for 1d coords and 1d sample
#------------------------------------------------------
def fmod(prob, coords):
    exits      = makeExits_1dsample(prob.sample, prob.probe, coords)
    diffAmps   = np.abs(bg.fftn(exits)) 
    return np.sum((diffAmps - prob.diffAmps)**2 * prob.mask) 

def Pmod_hat_diffs(diffAmps, psis, mask = None, alpha = 1.0e-10):
    if mask == None :
        mask = np.ones_like(diffAmps[0], dtype=np.bool)
    exits_out = psis
    exits_out = exits_out * (mask * diffAmps / np.clip(np.abs(exits_out), alpha, np.inf) \
                   + (~mask) )
    return exits_out

def emod_grad_coords_1d(T, probe, coords, diffAmps):
    exits_d = bg.fftn(makeExits_grad_1d(T, probe, coords, coords_d = None))
    exits   = bg.fftn(makeExits_1dsample(T, probe, coords))
    exits   = np.conj(exits_d) * (exits - Pmod_hat_diffs(diffAmps, exits))
    out     = np.sum(np.real(exits), axis = 2)
    return 2.0 * np.sum(out, axis = 1)

def emod_grad_dot_coords_1d(coords_d, T, probe, coords, diffAmps):
    exits_d = bg.fftn(makeExits_grad_1d(T, probe, coords, coords_d))
    exits   = bg.fftn(makeExits_1dsample(T, probe, coords))
    exits   = np.conj(exits_d) * (exits - Pmod_hat_diffs(diffAmps, exits))
    return 2 * np.sum(np.real(exits) )

def sample_grad_trans_1d(T, coord, coord_d):
    """Calculate : F-1[ Ti_hat (-2 pi i / Nx) Ri_d n ]"""
    Nx       = T.shape[-1]
    x        = bg.make_xy([Nx], origin=(0,0))
    #
    array    = np.zeros_like(T)
    array[:] = -2.0J * np.pi * (coord_d[1] * x / float(Nx)) * np.exp(-2.0J * np.pi * (coord[1] * x / float(Nx)))
    return bg.ifftn_1d(bg.fftn_1d(T) * array)

def makeExits_grad_1d_old(T, probe, coords, coords_d):
    """Calculate the exit surface waves but with T_i = sample_grad_trans_1d(T, coord, coord_d)
    
    if coords_d == None then T_i = sample_grad_trans_1d(T, coord, I)"""
    exits = np.zeros((len(coords), probe.shape[0], probe.shape[1]), dtype=np.complex128)
    #
    if np.all(coords_d == None):
        I = np.ones_like(coords[0])
        s_calc = False
    else :
        s_calc = True
    #
    for i in range(len(coords)):
        if s_calc :
            T_g      = sample_grad_trans_1d(T, coords[i], coords_d[i])
        else :
            T_g      = sample_grad_trans_1d(T, coords[i], I)
        exits[i] = T_g[:probe.shape[0], :probe.shape[1]]
    exits *= probe 
    return exits

def makeExits_grad_1d(sample, probe, coords, coords_d):
    sample1d        = sample[0, :]
    sample_stack    = np.zeros((len(coords), sample1d.shape[0]), dtype=sample1d.dtype)
    sample_stack[:] = sample1d
    # 
    # make the phase ramp for each 1d sample in the stack
    x_stack = np.zeros_like(sample_stack)
    x       = bg.make_xy([sample1d.shape[0]], origin=(0,))
    x       = -2.0J * np.pi * (x / float(sample1d.shape[0]))
    if np.all(coords_d == None ):
        for i, coord in enumerate(coords):
            x_stack[i] = np.exp(x * coords[i][1])
    else :
        for i, coord in enumerate(coords):
            x_stack[i] = np.exp(x * coords[i][1]) * coords_d[i][1]  
    x_stack = x_stack * x 
    #
    # shift the whole sample stack
    sample_stack  = bg.fftn_1d(sample_stack)
    sample_stack  = sample_stack * x_stack
    sample_stack  = bg.ifftn_1d(sample_stack)
    sample_stack  = sample_stack[:, :probe.shape[1]]
    #
    # make the exit waves
    exits = np.zeros((len(coords), probe.shape[0], probe.shape[1]), dtype=np.complex128)
    for i in range(len(coords)):
        exits[i, :] = sample_stack[i]  
    return exits * probe

def makeExits2(sample, probe, coords, exits):
    """Calculate the exit surface waves with no wrapping and assuming integer coordinate shifts"""
    for i, coord in enumerate(coords):
        exits[i] = sample[-coord[0]:probe.shape[0]-coord[0], -coord[1]:probe.shape[1]-coord[1]]
    exits *= probe 
    return exits

def makeExits_old(sample, probe, coords):
    """Calculate the exit surface waves with possible wrapping using the Fourier shift theorem"""
    exits = np.zeros((len(coords), probe.shape[0], probe.shape[1]), dtype=np.complex128)
    for i in range(len(coords)):
        exits[i] = bg.roll(sample, coords[i])[:probe.shape[0], :probe.shape[1]]
    exits *= probe 
    return exits

#--------------------------------------------------
#  Warning !!!! this is temorary for 1dsample
#--------------------------------------------------
def makeExits(sample, probe, coords):
    sample1d        = sample[0, :]
    sample_stack    = np.zeros((len(coords), sample1d.shape[0]), dtype=sample1d.dtype)
    sample_stack[:] = sample1d
    # 
    # make the phase ramp for each 1d sample in the stack
    x_stack = np.zeros_like(sample_stack)
    x       = bg.make_xy([sample1d.shape[0]], origin=(0,))
    x       = -2.0J * np.pi * (x / float(sample1d.shape[0]))
    for i, coord in enumerate(coords):
        x_stack[i] = x * coord[1]
    #
    # shift the whole sample stack
    sample_stack  = bg.fftn_1d(sample_stack)
    sample_stack  = sample_stack * np.exp(x_stack)
    sample_stack  = bg.ifftn_1d(sample_stack)
    sample_stack  = sample_stack[:, :probe.shape[1]]
    #
    # make the exit waves
    exits = np.zeros((len(coords), probe.shape[0], probe.shape[1]), dtype=np.complex128)
    for i in range(len(coords)):
        exits[i, :] = sample_stack[i]  
    return exits * probe

def makeExits_1dsample(sample, probe, coords):
    sample1d        = sample[0, :]
    sample_stack    = np.zeros((len(coords), sample1d.shape[0]), dtype=sample1d.dtype)
    sample_stack[:] = sample1d
    # 
    # make the phase ramp for each 1d sample in the stack
    x_stack = np.zeros_like(sample_stack)
    x       = bg.make_xy([sample1d.shape[0]], origin=(0,))
    x       = -2.0J * np.pi * (x / float(sample1d.shape[0]))
    for i, coord in enumerate(coords):
        x_stack[i] = x * coord[1]
    #
    # shift the whole sample stack
    sample_stack  = bg.fftn_1d(sample_stack)
    sample_stack  = sample_stack * np.exp(x_stack)
    sample_stack  = bg.ifftn_1d(sample_stack)
    sample_stack  = sample_stack[:, :probe.shape[1]]
    #
    # make the exit waves
    exits = np.zeros((len(coords), probe.shape[0], probe.shape[1]), dtype=np.complex128)
    for i in range(len(coords)):
        exits[i, :] = sample_stack[i]  
    return exits * probe

def input_output(inputDir):
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
    sequence.txt                list of algorithms to use (in order). 
                                e.g.
                                HIO_sample = 100
                                ERA_sample = 100
                                HIO_probe  = 50
                                ERA_probe  = 50
                                # and so on
    """
    # Does the directory exist? Do the files exist? This will be handled by bg.binary_in...
    print 'Loading the diffraction data...'
    diffs = bg.binary_in(inputDir + 'diffs', dt=np.float64, dimFnam=True)
    #
    shape = diffs.shape
    #
    # load the y,x pixel shift coordinates
    print 'Loading the ij coordinates...'
    coords   = bg.binary_in(inputDir + 'coordsInit', dt=np.float64, dimFnam=True)
    print 'warning: casting the coordinates from float to ints.'
    coords = np.array(coords, dtype=np.int32)
    #
    # load the mask
    if fnamBase_match(inputDir + 'mask'):
        print 'Loading the mask...'
        mask = bg.binary_in(inputDir + 'mask', dt=np.float64, dimFnam=True)
        mask = np.array(mask, dtype=np.bool)
    else :
        print 'no mask...'
        mask = np.ones_like(diffAmps[0], dtype=np.bool)
    #
    # load the probe
    if fnamBase_match(inputDir + 'probeInit'):
        print 'Loading the probe...'
        probe = bg.binary_in(inputDir + 'probeInit', dt=np.complex128, dimFnam=True)
    else :
        print 'generating a random probe...'
        probe = np.random.random(shape) + 1J * np.random.random(shape) 
    #
    # load the sample
    if fnamBase_match(inputDir + 'sampleInit'):
        print 'Loading the sample...'
        sample = bg.binary_in(inputDir + 'sampleInit', dt=np.complex128, dimFnam=True)
    else :
        print 'generating a random sample...'
        sample = np.random.random(shape) + 1J * np.random.random(shape) 
    #
    # load the sample support
    if fnamBase_match(inputDir + 'sample_support'):
        print 'Loading the sample support...'
        sample_support = bg.binary_in(inputDir + 'sample_support', dt=np.float64, dimFnam=True)
    else :
        sample_support = np.ones_like(sample)
    sample_support = np.array(sample_support, dtype=np.bool)
    #
    # load the sequence, cannot be a dict! This screws up the order
    print 'Loading the sequence file...'
    f = open(inputDir + 'sequence.txt', 'r')
    sequence = []
    for line in f:
        temp = line.rsplit()
        if len(temp) == 3 :
            if temp[0][0] != '#':
                if temp[1] == '=':
                    sequence.append([temp[0], temp[2]])
    #
    # If the sample is 1d then do a 1d retrieval 
    if len(sample.shape) == 1 and GPU_calc == False :
        print '1d sample => 1d Ptychography'
        prob = Ptychography_1dsample(diffs, coords, mask, probe, sample, sample_support)
        #
    elif len(sample.shape) == 2 and GPU_calc == False :
        print '2d sample => 2d Ptychography'
        prob = Ptychography(diffs, coords, mask, probe, sample, sample_support)
        #
    elif len(sample.shape) == 2 and GPU_calc == True :
        print 'Performing calculations on GPU'
        print '2d sample => 2d Ptychography'
        prob = Ptychography_gpu(diffs, coords, mask, probe, sample, sample_support)
    elif len(sample.shape) == 1 and GPU_calc == True :
        print 'Performing calculations on GPU'
        print '1d sample => 1d Ptychography'
        prob = Ptychography_1dsample_gpu(diffs, coords, mask, probe, sample, sample_support)
        #
    return prob, sequence

def runSequence(prob, sequence):
    if isinstance(prob, Ptychography)==False and isinstance(prob, Ptychography_gpu)==False :
        raise ValueError('prob must be an instance of the class Ptychography')
    #
    # Check the sequence list
    run_seq = []
    for i in range(len(sequence)):
        if sequence[i][0] in ('Pmod_probe', 'ERA_sample', 'ERA_probe', 'ERA_both', 'HIO_sample', 'HIO_probe', 'back_prop', 'Thibault_sample', 'Thibault_probe', 'Thibault_both', 'Thibault_both_av', 'Huang', 'coords_update_1d'):
            # This will return an error if the string is not formatted properly (i.e. as an int)
            if sequence[i][0] == 'ERA_sample':
                run_seq.append(sequence[i] + [prob.ERA_sample])
            #
            if sequence[i][0] == 'Pmod_probe':
                run_seq.append(sequence[i] + [prob.Pmod_probe])
            #
            if sequence[i][0] == 'ERA_probe':
                run_seq.append(sequence[i] + [prob.ERA_probe])
            #
            if sequence[i][0] == 'ERA_both':
                run_seq.append(sequence[i] + [prob.ERA_both])
            #
            if sequence[i][0] == 'HIO_sample':
                run_seq.append(sequence[i] + [prob.HIO_sample])
            #
            if sequence[i][0] == 'HIO_probe':
                run_seq.append(sequence[i] + [prob.HIO_probe])
            #
            if sequence[i][0] == 'Thibault_sample':
                run_seq.append(sequence[i] + [prob.Thibault_sample])
            #
            if sequence[i][0] == 'Thibault_probe':
                run_seq.append(sequence[i] + [prob.Thibault_probe])
            #
            if sequence[i][0] == 'Thibault_both':
                run_seq.append(sequence[i] + [prob.Thibault_both])
            #
            if sequence[i][0] == 'Thibault_both_av':
                run_seq.append(sequence[i] + [prob.Thibault_both_av])
            #
            if sequence[i][0] == 'Huang':
                run_seq.append(sequence[i] + [prob.Huang])
            #
            if sequence[i][0] == 'coords_update_1d':
                run_seq.append(sequence[i] + [prob.coords_update_1d])
            #
            if sequence[i][0] == 'back_prop':
                run_seq.append(sequence[i] + [prob.back_prop])
            #
            run_seq[-1][1] = int(sequence[i][1])

        elif sequence[i][0] in ('pmod_int'):
            if sequence[i][0] == 'pmod_int':
                if sequence[i][1] == 'True':
                    print 'exluding the values of sqrt(I) that fall in the range (0 --> 1)'
                    prob.pmod_int = True
        else :
            raise NameError("What algorithm is this?! I\'ll tell you one thing, it is not part of : 'ERA_sample', 'ERA_probe', 'ERA_both', 'HIO_sample', 'HIO_probe', 'back_prop', 'Thibault_sample', 'Thibault_probe', 'Thibault_both', 'Huang' " + sequence[i][0])
    #
    for seq in run_seq:
        print 'Running ', seq[0]
        seq[2](iters = seq[1])
    #
    return prob

if __name__ == '__main__':
    print '#########################################################'
    print 'Ptychography routine'
    print '#########################################################'
    inputdir, outputdir = main(sys.argv[1:])
    print 'input directory is ', inputdir
    print 'output directory is ', outputdir
    prob, sequence = input_output(inputdir)
    prob           = runSequence(prob, sequence)
    print ''
    #
    # output the results
    bg.binary_out(prob.sample, outputdir + 'sample_retrieved', dt=np.complex128, appendDim=True)
    bg.binary_out(prob.probe, outputdir + 'probe_retrieved', dt=np.complex128, appendDim=True)
    bg.binary_out(prob.coords, outputdir + 'coords_retrieved', appendDim=True)
    #
    bg.binary_out(np.array(prob.error_mod), outputdir + 'error_mod', appendDim=True)
    bg.binary_out(np.array(prob.error_sup), outputdir + 'error_sup', appendDim=True)
    if len(prob.error_conv) > 0 :
        bg.binary_out(np.array(prob.error_conv), outputdir + 'error_conv', appendDim=True)
    #
    # do not output this because they can easily be generated from the retrieved sample, probe and coords
    # bg.binary_out(np.abs(bg.fftn(prob.exits))**2, outputDir + 'diffs_retrieved', appendDim=True)

    
