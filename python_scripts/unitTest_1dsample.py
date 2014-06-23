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
shape_sample = (32, 128)
amp          = bg.scale(bg.brog(shape_sample), 0.0, 1.0)
phase        = bg.scale(bg.twain(shape_sample), -np.pi, np.pi)
sample       = amp * np.exp(1J * phase)

# Now let's just take a slice out of the sample
sample_1d = sample[sample.shape[0]/2, :]

# and expand back to the sample
sample[:] = sample_1d

# Make an illumination on the data grid
shape_illum = (32, 64)
probe       = bg.circle_new(shape_illum, radius=0.5, origin=[shape_illum[0]/2-1, shape_illum[1]/2 - 1]) + 0J

# Make sample coordinates (y, x)
# these will be relative shift coordinates in pixel units
# positive numbers indicate that the sample is shifted in the positive axis directions
# so sample_shifted = sample(y - yi, x - xi)
# These will be a list of [y, x]

dx = 5
dy = shape_sample[0]
x, y    = np.meshgrid(  range(3, shape_sample[1] - probe.shape[1] - 3, dx), range(0, shape_sample[0], dy))
coords0 = zip(y.flatten(), x.flatten())
coords0 = np.array(coords0)
coords0 = -np.array(coords0)
print coords0.shape
#
# add a random offset of three pixels in x or y
dcoords  = np.random.random(coords0.shape) * 6 - 3
dcoords[:, 0] = 0
print dcoords.shape
coords  = coords0 + np.array(dcoords, dtype=np.int32)
# only add random stuff to the x-direction
coords[:,0] = coords0[:, 0]

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
    print 'processing coordinate', coord
    exitF = bg.fft2(makeExit(sample, probe, coord))
    diffs.append(np.abs(exitF)**2)

print 'making heatmap'
heatmap = np.zeros_like(sample)
for coord in coords:
    temp    = np.zeros_like(sample)
    temp[:probe.shape[0], :probe.shape[1]] = makeExit(np.ones_like(sample), probe, coord)
    temp = bg.roll(temp, -coord)
    heatmap += np.abs(temp)**2

sampleInit = np.random.random(sample_1d.shape) + 1J*np.random.random(sample_1d.shape)
#sampleInit = sample
#probeInit = np.random.random((shape_illum)) + 1J*np.random.random((shape_illum))
#probeInit  = bg.circle_new(shape_illum, radius=0.3, origin=[shape_illum[0]/2-1, shape_illum[1]/2 - 1]) + 0J
probeInit  = probe

# Output 
outputdir = main(sys.argv[1:])
print 'outputputing files...'
print 'output directory is ', outputdir

sequence = """# This is a sequence file which determines the ptychography algorithm to use
ERA_sample = 1000
"""
#Thibault_both = 300

with open(outputdir + "sequence.txt", "w") as text_file:
    text_file.write(sequence)

bg.binary_out(probe, outputdir + 'probe', dt=np.complex128, appendDim=True)
bg.binary_out(sample_1d, outputdir + 'sample', dt=np.complex128, appendDim=True)
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

sample_ret = bg.binary_in(outputdir + 'sample_retrieved', dt=np.complex128, dimFnam=True)
c = np.sum(np.conj(sample_ret) * sample) / np.sum(np.abs(sample_ret)**2 + 1.0e-10)
print 'sample error', bg.l2norm(c * sample_ret, sample)

mask = (heatmap > 1.0e-1 * heatmap.max())
print 'mask area' , np.sum(mask)

sample_ret_m = sample_ret * mask
sample_m = sample * mask
c = np.sum(np.conj(sample_ret_m) * sample_m) / np.sum(np.abs(sample_ret_m)**2 + 1.0e-10)
print 'sample error', bg.l2norm(c * sample_ret_m, sample_m)
