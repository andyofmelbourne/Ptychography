import numpy as np
import sys
from ctypes import *
import bagOfns as bg
from utility_Ptych import makeExits


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
            self.update_progress(i / max(1.0, float(iters-1)), 'ERA sample', i, self.error_mod[-1], self.error_sup[-1])
        return self

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
            self.update_progress(i / max(1.0, float(iters-1)), 'ERA Probe', i, self.error_mod[-1], self.error_sup[-1])
        return self

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
            self.update_progress(i / max(1.0, float(iters-1)), 'ERA both', i, self.error_mod[-1], self.error_sup[-1])
        return self

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
            self.update_progress(i / max(1.0, float(iters-1)), 'HIO sample', i, self.error_mod[-1], self.error_sup[-1])
            #
            self.exits = exits
        return self

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
            self.update_progress(i / max(1.0, float(iters-1)), 'HIO probe', i, self.error_mod[-1], self.error_sup[-1])
            #
            self.exits = exits
        return self

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
            self.update_progress(i / max(1.0, float(iters-1)), 'Thibault sample', i, self.error_conv[-1], self.error_sup[-1])
        return self

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
            self.update_progress(i / max(1.0, float(iters-1)), 'Thibault probe', i, self.error_conv[-1], self.error_sup[-1])
        return self

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
            self.update_progress(i / max(1.0, float(iters-1)), 'Thibault sample / probe', i, self.error_conv[-1], self.error_sup[-1])
        return self

    def Huang(self, iters=None):
        """This is the algorithm used in "11 nm Hard X-ray focus from a large-aperture multilayer Laue lens" (2013) nature"""
        def randsample():
            array = np.random.random(self.sample.shape) + 1J * np.random.random(self.sample.shape) 
            return array
        #
        for i in range(3):
            self.sample     = randsample()
            self.sample_sum = None
            self.exits = makeExits(self.sample, self.probe, self.coords)
            #
            self.Thibault_sample(iters=5)
            #
            self.Thibault_both(iters=10)
            #
            probes = []
            for j in range(10):
                self.Thibault_both(iters=1)
                probes.append(self.probe.copy())
            self.probe     = np.sum(np.array(probes), axis=0) / float(len(probes))
            self.probe_sum = None
        #
        probe  = self.probe.copy()
        probes  = []
        samples = []
        for i in range(3):
            self.sample     = randsample()
            self.sample_sum = None
            self.probe      = probe.copy()
            self.probe_sum  = None
            self.exits = makeExits(self.sample, self.probe, self.coords)
            #
            self.Thibault_sample(iters=5)
            #
            self.Thibault_both(iters=5)
            #
            for j in range(5):
                self.Thibault_both(iters=1)
                probes.append(self.probe.copy())
                samples.append(self.sample.copy())
        self.probe = np.sum(np.array(probes), axis=0) / float(len(probes))
        self.sample = np.sum(np.array(samples), axis=0) / float(len(samples))
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
        return self
    
    def update_progress(self, progress, algorithm, i, emod, esup):
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



def forward_sim():
    # sample
    shape_sample = (80, 180)
    amp          = bg.scale(bg.brog(shape_sample), 0.0, 1.0)
    phase        = bg.scale(bg.twain(shape_sample), -np.pi, np.pi)
    sample       = amp * np.exp(1J * phase)
    sample_support = np.ones_like(sample, dtype=np.bool)

    shape_sample = (128, 256)
    sample         = bg.zero_pad(sample,         shape_sample, fillvalue=1.0)
    sample_support = bg.zero_pad(sample_support, shape_sample)
    
    # probe
    shape_illum = (64, 128)
    probe       = bg.circle_new(shape_illum, radius=0.5, origin='centre') + 0J
        
    # make some sample positions
    xs = range(shape_illum[1] - shape_sample[1], 1, 4)
    ys = range(shape_illum[0] - shape_sample[0], 1, 4)
    xs, ys = np.meshgrid( xs, ys )
    coords = zip(ys.ravel(), xs.ravel())

    # diffraction patterns
    diffs = makeExits(sample, probe, coords)
    diffs = np.abs(bg.fft2(diffs))**2

    mask = np.ones_like(diffs[0], dtype=np.bool)
    return diffs, coords, mask, probe, sample, sample_support



# Example usage
if __name__ == '__main__' :
    diffs, coords, mask, probe, sample, sample_support = forward_sim()

    sample0 = np.random.random(sample.shape) + 1J*np.random.random(sample.shape)

    prob = Ptychography(diffs, coords, mask, probe, sample0, sample_support) 
    
    # do 100 ERA iterations
    prob = Ptychography.ERA_sample(prob, 100)
    
    # check the fidelity inside of the illuminated region:
    probe_mask = prob.probe_sum > prob.probe_sum.max() * 1.0e-10
    
    print '\nfidelity: ', bg.l2norm(sample * probe_mask, prob.sample * probe_mask)
