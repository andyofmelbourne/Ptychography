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

print 'GPU_calc', GPU_calc

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
        if sequence[i][0] in ('ERA_sample', 'ERA_probe', 'ERA_both', 'HIO_sample', 'HIO_probe', 'back_prop', 'Thibault_sample', 'Thibault_probe', 'Thibault_both', 'Huang', 'coords_update_1d'):
            # This will return an error if the string is not formatted properly (i.e. as an int)
            if sequence[i][0] == 'ERA_sample':
                run_seq.append(sequence[i] + [prob.ERA_sample])
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

    
