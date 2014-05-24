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
shape_sample = (64, 128)
amp   = bg.scale(bg.brog(shape_sample), 0.0, 1.0)
phase = bg.scale(bg.twain(shape_sample), -np.pi, np.pi)
sample = amp * np.exp(1J * phase)

# Make an illumination on the data grid
shape_illum = (32, 64)
probe       = bg.circle_new(shape_illum, radius=0.5, origin=[shape_illum[0]/2-1, shape_illum[1]/2 - 1]) + 0J

# Make sample coordinates (y, x)
# these will be relative shift coordinates in pixel units
# positive numbers indicate that the sample is shifted in the positive axis directions
# so sample_shifted = sample(y - yi, x - xi)
# These will be a list of [y, x]

dx = dy = 10
y, x    = np.meshgrid( range(0, shape_sample[0], dy), range(0, shape_sample[1], dx), indexing='ij' )
coords0 = zip(y.flatten(), x.flatten())
coords0 = np.array(coords0)
print coords0.shape
#
# add a random offset of three pixels in x or y
dcoords  = np.random.random(coords0.shape) * 6 - 3
print dcoords.shape
coords  = coords0 + np.array(dcoords, dtype=np.int32)

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
    print 'processing coordinate', coord
    exitF = bg.fft2(makeExit(sample, probe, coord))
    diffs.append(np.abs(exitF)**2)

# This ensures reproducibility
np.random.seed(1)
sampleInit = np.random.random((shape_sample)) + 1J*np.random.random((shape_sample))
#sampleInit = sample
#probeInit = np.random.random((shape_illum)) + 1J*np.random.random((shape_illum))
#probeInit  = bg.circle_new(shape_illum, radius=0.3, origin=[shape_illum[0]/2-1, shape_illum[1]/2 - 1]) + 0J
probeInit  = probe

# Output 
outputdir = main(sys.argv[1:])
print 'outputputing files...'
print 'output directory is ', outputdir

sequence = """# This is a sequence file which determines the ptychography algorithm to use
Thibault_sample = 200
ERA_sample = 500
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


# run the job
subprocess.Popen([sys.executable, 'Ptychography.py', '-i', outputdir, '-o', outputdir])
