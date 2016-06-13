import numpy as np
import sys
from ctypes import *
import bagOfns as bg
from utility_Ptych import makeExits3 as makeExits
from utility_Ptych import ERA, HIO, Thibault, update_progress


class Ptychography(object):
    def __init__(self, diffs, coords, probe, sample, mask = None, sample_support = None, pmod_int = False): 
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
        if mask is None :
            self.mask = np.ones_like(diffs[0], dtype=np.bool)
        else :
            self.mask       = bg.quadshift(mask)
        self.probe      = probe
        self.sample     = sample
        self.alpha_div  = 1.0e-10
        self.eMod  = []
        self.eSup  = []
        self.error_conv = []
        self.probe_sum  = None
        self.sample_sum = None
        self.diffNorm   = np.sum(self.mask * (self.diffAmps)**2)
        self.pmod_int   = pmod_int
        if sample_support is None :
            self.sample_support = np.ones_like(sample, dtype=np.bool)
        else :
            self.sample_support = sample_support
        self.iteration  = 0


    def ERA(self, iters=1, update = 'sample'):
        """Caculate the update for the exit surface waves using the Error Reduction Algorithm

        We can specify the one of 'sample', 'probe' or 'both' should be updated.
        """
        if update == 'sample' :
            Psup = lambda ex : makeExits(self.Psup_sample(ex, inPlace=True), self.probe, self.coords) 
            alg = 'ERA sample'
        
        elif update == 'probe' :
            Psup = lambda ex : makeExits(self.sample, self.Psup_probe(ex, inPlace=True), self.coords) 
            alg = 'ERA probe'
        
        elif update == 'both':
            temp_sample = lambda ex : makeExits(self.Psup_sample(ex, inPlace=True, thresh=1.0), self.probe, self.coords)
            temp_probe  = lambda ex : makeExits(self.sample, self.Psup_probe(ex, inPlace=True), self.coords) 
            Psup = lambda ex : temp_probe(temp_sample(ex))
            alg = 'ERA both'
        
        print 'i, eMod, eSup'
        for i in range(iters):
            self.exits = ERA(self.exits, self.Pmod, Psup)
            
            # calculate errors every 4 iters
            if (i % 4 == 0) or (i == iters - 1) :
                eMod = self.calc_eMod(self.exits)
                eSup = self.calc_eSup(self.exits, Psup)
                self.eMod.append([self.iteration, eMod])
                self.eSup.append([self.iteration, eSup])
            
            self.iteration += 1
             
            # display errors
            update_progress(i / max(1.0, float(iters-1)), alg, i, self.eMod[-1][1], self.eSup[-1][1])
        return self

    def HIO(self, iters=1, beta=1., update='sample'):
        """Caculate the update for the exit surface waves using the Hybrid Input Output algorithm

        We can specify the one of 'sample', 'probe' or 'both' should be updated.
        """
        if update == 'sample' :
            Psup = lambda ex : makeExits(self.Psup_sample(ex, inPlace=True), self.probe, self.coords) 
            alg = 'HIO sample'
        
        elif update == 'probe' :
            Psup = lambda ex : makeExits(self.sample, self.Psup_probe(ex, inPlace=True), self.coords) 
            alg = 'HIO probe'
        
        elif update == 'both':
            temp_sample = lambda ex : makeExits(self.Psup_sample(ex, inPlace=True, thresh=1.0), self.probe, self.coords)
            temp_probe  = lambda ex : makeExits(self.sample, self.Psup_probe(ex, inPlace=True), self.coords) 
            Psup = lambda ex : temp_probe(temp_sample(ex))
            alg = 'HIO both'
        
        print 'i, eMod, eSup'
        for i in range(iters):
            self.exits = HIO(self.exits, self.Pmod, Psup, beta=beta)
            
            # calculate errors every 4 iters
            if (i % 4 == 0) or (i == iters - 1) :
                eMod = self.calc_eMod(self.exits)
                eSup = self.calc_eSup(self.exits, Psup)
                self.eMod.append([self.iteration, eMod])
                self.eSup.append([self.iteration, eSup])
            
            self.iteration += 1
             
            # display errors
            update_progress(i / max(1.0, float(iters-1)), alg, i, self.eMod[-1][1], self.eSup[-1][1])
        return self

    def Thibault(self, iters=1, beta=1., update='sample'):
        """Caculate the update for the exit surface waves using Thibault's application of the difference map algorithm

        We can specify the one of 'sample', 'probe' or 'both' should be updated.
        """
        if update == 'sample' :
            Psup = lambda ex : makeExits(self.Psup_sample(ex, inPlace=True), self.probe, self.coords) 
            alg = 'DM sample'
        
        elif update == 'probe' :
            Psup = lambda ex : makeExits(self.sample, self.Psup_probe(ex, inPlace=True), self.coords) 
            alg = 'DM probe'
        
        elif update == 'both':
            temp_sample = lambda ex : makeExits(self.Psup_sample(ex, inPlace=True, thresh=1.0), self.probe, self.coords)
            temp_probe  = lambda ex : makeExits(self.sample, self.Psup_probe(ex, inPlace=True), self.coords) 
            Psup = lambda ex : temp_probe(temp_sample(ex))
            alg = 'DM both'
        
        print 'i, eMod, eSup'
        for i in range(iters):
            self.exits = Thibault(self.exits, self.Pmod, Psup)
            
            # calculate errors every 4 iters
            if (i % 4 == 0) or (i == iters - 1) :
                eMod = self.calc_eMod(self.exits)
                eSup = self.calc_eSup(self.exits, Psup)
                self.eMod.append([self.iteration, eMod])
                self.eSup.append([self.iteration, eSup])
            
            self.iteration += 1
             
            # display errors
            update_progress(i / max(1.0, float(iters-1)), alg, i, self.eMod[-1][1], self.eSup[-1][1])
        return self

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
          
        # Calculate denominator
        # but only do this if it hasn't been done already
        # (we must set self.probe_sum = None when the probe/coords has changed)
        if self.probe_sum is None :
            self.probe_sum = self.heatmap().copy()
          
        # Calculate numerator
        exits2     = np.conj(self.probe) * exits 
        for coord, exit in zip(self.coords, exits2):
            sample_out[-coord[0]:self.shape[0]-coord[0], -coord[1]:self.shape[1]-coord[1]] += exit
         
        # divide
        sample_out  = sample_out / (self.probe_sum + self.alpha_div)
        sample_out  = sample_out * self.sample_support + ~self.sample_support
         
        if thresh :
            sample_out = bg.threshold(sample_out, thresh=thresh)
         
        if inPlace :
            self.sample = sample_out
        self.sample_sum = None
          
        return sample_out
    
    def Psup_probe(self, exits, inPlace=True):
        """ """
        probe_out  = np.zeros_like(self.probe)
        # 
        # Calculate denominator
        # but only do this if it hasn't been done already
        # (we must set self.sample_sum = None when the sample/coords has changed)
        if self.sample_sum is None :
            sample_sum  = np.zeros_like(self.sample, dtype=np.float64)
            temp        = np.real(self.sample * np.conj(self.sample))
            for coord in self.coords:
                sample_sum  += bg.roll(temp, coord)
            self.sample_sum = sample_sum[:self.shape[0], :self.shape[1]]
          
        # Calculate numerator
        # probe = sample * [sum np.conj(sample_shifted) * exit_shifted] / sum |sample_shifted|^2 
        sample_conj = np.conj(self.sample)
        for exit, coord in zip(exits, self.coords):
            probe_out += exit * sample_conj[-coord[0]:self.shape[0]-coord[0], -coord[1]:self.shape[1]-coord[1]]
         
        # divide
        probe_out   = probe_out / (self.sample_sum + self.alpha_div)
        if inPlace:
            self.probe = probe_out
        self.probe_sum = None
        # 
        return probe_out

    def heatmap(self):
        probe_sum   = np.zeros_like(self.sample, dtype=np.float64)
        probe_large = np.zeros_like(self.sample, dtype=np.float64)
        probe_large[:self.shape[0], :self.shape[1]] = np.real(np.conj(self.probe) * self.probe)
        for coord in self.coords:
            probe_sum  += bg.roll(probe_large, [-coord[0], -coord[1]])
        return probe_sum

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
        return self
    
    def calc_eMod(self, exits):
        eMod = self.mask * (np.abs(bg.fftn(exits)) - self.diffAmps)**2
        eMod = np.sum(eMod) / self.diffNorm
        eMod = np.sqrt(eMod)
        return eMod
    
    def calc_eSup(self, exits, Psup):
        eSup = np.abs( exits - Psup(exits) )**2
        eSup = np.sum(eSup) / self.diffNorm
        eSup = np.sqrt(eSup)
        return eSup


def forward_sim(shape_sample = (128, 256), shape_illum = (64, 128), nx = 8, ny = 8):
    # sample
    shape_sample0 = shape_sample
    amp          = bg.scale(bg.brog(shape_sample0), 0.0, 1.0)
    phase        = bg.scale(bg.twain(shape_sample0), -np.pi, np.pi)
    sample       = amp * np.exp(1J * phase)
    sample_support = np.ones_like(sample, dtype=np.bool)

    sample         = bg.zero_pad(sample,         shape_sample, fillvalue=1.0)
    sample_support = bg.zero_pad(sample_support, shape_sample)
    
    # probe
    probe       = bg.circle_new(shape_illum, radius=0.5, origin='centre') + 0J
        
    # make some sample positions
    xs = range(shape_illum[1] - shape_sample[1], 1, nx)
    ys = range(shape_illum[0] - shape_sample[0], 1, ny)
    xs, ys = np.meshgrid( xs, ys )
    coords = np.array(zip(ys.ravel(), xs.ravel()))

    # random offset 
    dcoords = (np.random.random(coords.shape) * 3).astype(np.int)
    coords += dcoords
    coords[np.where(coords > 0)] = 0
    coords[:, 0][np.where(coords[:, 0] < shape_illum[0] - shape_sample[0])] = shape_illum[0] - shape_sample[0]
    coords[:, 1][np.where(coords[:, 1] < shape_illum[1] - shape_sample[1])] = shape_illum[1] - shape_sample[1]

    # diffraction patterns
    diffs = makeExits(sample, probe, coords)
    diffs = np.abs(bg.fft2(diffs))**2

    mask = np.ones_like(diffs[0], dtype=np.bool)
    return diffs, coords, mask, probe, sample, sample_support



# Example usage
if __name__ == '__main__' :
    diffs, coords, mask, probe, sample, sample_support = forward_sim()
    
    sample0 = np.random.random(sample.shape) + 1J*np.random.random(sample.shape)
    probe0  = probe + 0.1 * (np.random.random(probe.shape) + 1J*np.random.random(probe.shape))
    
    prob = Ptychography(diffs, coords, probe, sample0, mask, sample_support) 
    
    # do 50 ERA iterations
    prob = Ptychography.Thibault(prob, 20, update='sample')
    prob = Ptychography.ERA(prob, 50, update='sample')
    
    # check the fidelity inside of the illuminated region:
    probe_mask = prob.probe_sum > prob.probe_sum.max() * 1.0e-10
    
    print '\nfidelity: ', bg.l2norm(sample * probe_mask, prob.sample * probe_mask)
