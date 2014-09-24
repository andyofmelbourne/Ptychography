import numpy as np
import scipy as sp
from scipy import ndimage
import os, sys, getopt
from ctypes import *
import bagOfns as bg
import time

def makeExits(sample, probe, coords):
    """Calculate the exit surface waves with possible wrapping using the Fourier shift theorem"""
    exits = np.zeros((len(coords), probe.shape[0], probe.shape[1]), dtype=np.complex128)
    for i in range(len(coords)):
        exits[i] = bg.roll(sample, coords[i])[:probe.shape[0], :probe.shape[1]]
    exits *= probe 
    return exits
