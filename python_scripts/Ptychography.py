# This code is primarily designed to run from the command 
# line with a configuration file as input from the user.
import numpy as np
import scipy as sp
from scipy import ndimage
import os, sys, getopt
from ctypes import *
import bagOfns as bg
import time

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
    def __init__(self, diffs, coords, mask, probe, sample, pmod_int = False): 
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
            #

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
            #

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
            #

    def Pmod(self, exits, pmod_int = False):
        exits_out = bg.fftn(exits)
        if self.pmod_int or pmod_int :
            exits_out = exits_out * (self.mask * self.diffAmps * (self.diffAmps > 0.99) /  np.clip(np.abs(exits_out), self.alpha_div, np.inf) \
                           + (~self.mask) )
        else :
            exits_out = exits_out * (self.mask * self.diffAmps / np.clip(np.abs(exits_out), self.alpha_div, np.inf) \
                           + (~self.mask) )
        exits_out = bg.ifftn(exits_out)
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
            temp[:self.shape[0], :self.shape[1]] = exits2[i]
            sample_out += bg.roll(temp, [-self.coords[i][0], -self.coords[i][1]])
        #
        # divide
        sample_out  = sample_out / (self.probe_sum + self.alpha_div)
        if thresh :
            sample_out = bg.threshold(sample_out, thresh=thresh)
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

    def __init__(self, diffs, coords, mask, probe, sample_1d, pmod_int = False): 
        self.sample_1d = sample_1d
        sample = np.zeros((probe.shape[0], sample_1d.shape[0]), dtype = sample_1d.dtype)
        sample[:] = sample_1d.copy()
        Ptychography.__init__(self, diffs, coords, mask, probe, sample, pmod_int)

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

def makeExits(sample, probe, coords):
    exits = np.zeros((len(coords), probe.shape[0], probe.shape[1]), dtype=np.complex128)
    for i in range(len(coords)):
        exits[i] = bg.roll(sample, coords[i])[:probe.shape[0], :probe.shape[1]]
    exits *= probe 
    return exits



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
    coords   = bg.binary_in(inputDir + 'coords', dt=np.float64, dimFnam=True)
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
    if len(sample.shape) == 1 :
        print '1d sample => 1d Ptychography'
        prob = Ptychography_1dsample(diffs, coords, mask, probe, sample)
    elif len(sample.shape) == 2 :
        print '2d sample => 2d Ptychography'
        prob = Ptychography(diffs, coords, mask, probe, sample)
    return prob, sequence

def runSequence(prob, sequence):
    if not isinstance(prob, Ptychography):
        raise ValueError('prob must be an instance of the class Ptychography')
    #
    # Check the sequence list
    run_seq = []
    for i in range(len(sequence)):
        if sequence[i][0] in ('ERA_sample', 'ERA_probe', 'HIO_sample', 'HIO_probe', 'back_prop', 'Thibault_sample', 'Thibault_probe', 'Thibault_both'):
            # This will return an error if the string is not formatted properly (i.e. as an int)
            if sequence[i][0] == 'ERA_sample':
                run_seq.append(sequence[i] + [prob.ERA_sample])
            #
            if sequence[i][0] == 'ERA_probe':
                run_seq.append(sequence[i] + [prob.ERA_probe])
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
            raise NameError("What algorithm is this?! I\'ll tell you one thing, it is not part of 'ERA_sample', 'ERA_probe', 'HIO_sample', 'HIO_probe': " + sequence[i][0])
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
    #
    # do not output this because they can easily be generated from the retrieved sample, probe and coords
    # bg.binary_out(np.abs(bg.fftn(prob.exits))**2, outputDir + 'diffs_retrieved', appendDim=True)


    
