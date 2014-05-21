import numpy as np
import scipy as sp
from scipy import ndimage
import sys
import os
from ctypes import *
import bagOfns as bg

# Generate a unit test for Ptychography
# Aims:
# generate diffraction data
# generate diffraction mask
# output zero pixel
# output probe coordinates


# Make a sample on a large grid
shape_sample = (64, 128)
amp   = bg.scale(bg.brog(shape_sample), 0.0, 1.0)
phase = bg.scale(bg.twain(shape_sample), -np.pi, np.pi)
sample = amp * np.exp(1J * phase)

# Make an illumination on the data grid
shape_illum = (32, 64)
probe       = bg.circle(shape_illum, radius=0.5, origin=[shape_illum[0]/2-1, shape_illum[1]/2 - 1]) + 0J

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
# mask = np.array(bg.circle(shape_illum, radius=0.1), dtype=np.bool)
# mask = ~mask
# Just ones for now
mask = np.ones_like(probe, dtype=np.bool)

print 'making diffraction patterns'
diffs = []
for coord in coords:
    print 'processing coordinate', coord
    exitF = bg.fft2(makeExit(sample, probe, coord))
    diffs.append(np.abs(exitF)**2)

def makeHeatMap(sample, probe, coords):
    print 'making heatMap'
    heatMap      = np.zeros_like(sample, dtype=np.float64)
    sample_shift = np.zeros_like(sample, dtype=np.float64)
    for coord in coords:
        print 'processing coordinate', coord
        sample_shift[:probe.shape[0], :probe.shape[1]] = np.abs(probe)
        heatMap     += bg.roll(sample_shift, [-coord[0], -coord[1]])
    return heatMap

sampleInit = np.random.random((shape_sample)) + 1J*np.random.random((shape_sample))
#sampleInit = sample
#probeInit = np.random.random((shape_illum)) + 1J*np.random.random((shape_illum))
#probeInit  = bg.circle(shape_illum, radius=0.3, origin=[shape_illum[0]/2-1, shape_illum[1]/2 - 1]) + 0J
probeInit  = probe

# Output 
dir_out = './test_cases/unit_test_Ptychography/'
bg.binary_out(probe, dir_out + 'probe', dt=np.complex128, appendDim=True)
bg.binary_out(sample, dir_out + 'sample', dt=np.complex128, appendDim=True)
bg.binary_out(mask, dir_out + 'mask', dt=np.float64, appendDim=True)
bg.binary_out(np.array(diffs), dir_out + 'diffs', dt=np.float64, appendDim=True)
bg.binary_out(np.array(coords), dir_out + 'coords', dt=np.float64, appendDim=True)


bg.binary_out(probeInit, dir_out + 'probeInit', dt=np.complex128, appendDim=True)
bg.binary_out(sampleInit, dir_out + 'sampleInit', dt=np.complex128, appendDim=True)
