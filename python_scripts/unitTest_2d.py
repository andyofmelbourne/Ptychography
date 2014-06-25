import numpy as np
import scipy as sp
from scipy import ndimage
import sys, os, getopt
import subprocess 
from ctypes import *
import bagOfns as bg

# Generate a unit test for Ptychography
# Aims:
# generate diffraction data
# generate diffraction mask
# output zero pixel
# output probe coordinates

def main(argv):
    outputdir = './'
    try :
        opts, args = getopt.getopt(argv,"ho:",["outputdir="])
    except getopt.GetoptError:
        print 'python makeUnitTest.py -o <outputdir>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'python makeUnitTest.py -o <outputdir>'
            sys.exit()
        elif opt in ("-o", "--outputdir"):
            outputdir = arg
    return outputdir

# Make a sample on a large grid
shape_sample = (128, 256)
amp          = bg.scale(bg.brog(shape_sample), 0.0, 1.0)
phase        = bg.scale(bg.twain(shape_sample), -np.pi, np.pi)
sample       = amp * np.exp(1J * phase)

# Make an illumination on the data grid
shape_illum = (64, 128)
probe       = bg.circle_new(shape_illum, radius=0.5, origin=[shape_illum[0]/2-1, shape_illum[1]/2 - 1]) + 0J

# Make sample coordinates (y, x)
# these will be relative shift coordinates in pixel units
# positive numbers indicate that the sample is shifted in the positive axis directions
# so sample_shifted = sample(y - yi, x - xi)
# These will be a list of [y, x]

dx = dy = 5
x, y    = np.meshgrid(  range(3, shape_sample[1] - probe.shape[1] - 3, dx), range(3, shape_sample[0] - probe.shape[0] - 3, dy))
coords0 = zip(y.flatten(), x.flatten())
coords0 = -np.array(coords0)
print coords0.shape
#
# add a random offset of three pixels in x or y
dcoords  = np.random.random(coords0.shape) * 6 - 3
print dcoords.shape
coords  = coords0 + np.array(dcoords, dtype=np.int32)
print coords

#coords = []
#for y in range(0, shape_sample[0], dy):
#    for x in range(0, shape_sample[1], dx):
#        coords.append([y, x])

def makeExit(sample, probe, shift = [0,0]):
    sample_shift = bg.roll(sample, shift)
    exit = probe * sample_shift[:probe.shape[0], :probe.shape[1]]
    return exit

# Make a detector mask
# mask = np.array(bg.circle_new(shape_illum, radius=0.1), dtype=np.bool)
# mask = ~mask
# Just ones for now
mask = np.ones_like(probe, dtype=np.bool)

print 'making diffraction patterns'
diffs = []
for coord in coords:
    exitF = bg.fft2(makeExit(sample, probe, coord))
    diffs.append(np.abs(exitF)**2)

print 'making heatmap'
heatmap = np.zeros_like(sample)
for coord in coords:
    temp    = np.zeros_like(sample)
    temp[:probe.shape[0], :probe.shape[1]] = makeExit(np.ones_like(sample), probe, coord)
    temp = bg.roll(temp, -coord)
    heatmap += np.abs(temp)**2

sampleInit = np.random.random((shape_sample)) + 1J*np.random.random((shape_sample))
#sampleInit = sample
#probeInit = np.random.random((shape_illum)) + 1J*np.random.random((shape_illum))
probeInit  = bg.circle_new(shape_illum, radius=0.3, origin=[shape_illum[0]/2-1, shape_illum[1]/2 - 1]) + 0J
#probeInit  = probe

# Output 
outputdir = main(sys.argv[1:])
print 'outputputing files...'
print 'output directory is ', outputdir

sequence = """# This is a sequence file which determines the ptychography algorithm to use
Thibault_both = 500
ERA_both = 100
"""

with open(outputdir + "sequence.txt", "w") as text_file:
    text_file.write(sequence)

bg.binary_out(probe, outputdir + 'probe', dt=np.complex128, appendDim=True)
bg.binary_out(sample, outputdir + 'sample', dt=np.complex128, appendDim=True)
bg.binary_out(mask, outputdir + 'mask', dt=np.float64, appendDim=True)
bg.binary_out(np.array(diffs), outputdir + 'diffs', dt=np.float64, appendDim=True)
bg.binary_out(np.array(coords), outputdir + 'coords', dt=np.float64, appendDim=True)

bg.binary_out(probeInit, outputdir + 'probeInit', dt=np.complex128, appendDim=True)
bg.binary_out(sampleInit, outputdir + 'sampleInit', dt=np.complex128, appendDim=True)



print 'Finished!'
print ''
print 'Now run the test with:'
print 'python Ptychography.py -i', outputdir, ' -o',outputdir


# run Ptychography and time it using the bash command "time"
subprocess.call('time python Ptychography.py' + ' -i' + outputdir + ' -o' + outputdir, shell=True)

if False :
    def fidelity(o1 , o2):
        return np.sqrt(np.sum(np.abs(o1 - o2)**2) / np.sum(np.abs(o1)**2))

    def fidelity_cons(o1 , o2):
        c = np.sum(np.conj(o1) * o2) / np.sum(np.abs(o1)**2)
        return fidelity(c * o1, o2)

    def fidelity_shift(o1 , o2):
        errors = []
        for i in range(o1.shape[0]):
            for j in range(o1.shape[1]):
                o_shift = bg.roll(o1, [i, j])
                errors.append(fidelity(o_shift, o2))
        return np.array(errors).min()

    def fidelity_shift_cons(o1 , o2):
        errors = []
        for i in range(o1.shape[0]):
            for j in range(o1.shape[1]):
                o_shift = bg.roll(o1, [i, j])
                errors.append(fidelity_cons(o_shift, o2))
        return np.array(errors).min()

    def fidelity_shift_cons_mask(o1, masko1, o2):
        errors = []
        for i in range(o1.shape[0]):
            for j in range(o1.shape[1]):
                o_shift = bg.roll(o1, [i, j])
                m_shift = bg.roll(masko1, [i, j])
                errors.append(fidelity_cons(o_shift * m_shift, o2 * m_shift)) 
        return np.array(errors).min()


    sample_ret = bg.binary_in(outputdir + 'sample_retrieved', dt=np.complex128, dimFnam=True)
    probe_ret  = bg.binary_in(outputdir + 'probe_retrieved', dt=np.complex128, dimFnam=True)

    mask = (heatmap > 1.0e-1 * heatmap.max())
    print 'mask area' , np.sum(mask)

    print 'sample error shift and constant corrected:', fidelity_shift_cons_mask(sample, mask, sample_ret)
    print 'probe  error shift and constant corrected:', fidelity_shift_cons(probe, probe_ret)
