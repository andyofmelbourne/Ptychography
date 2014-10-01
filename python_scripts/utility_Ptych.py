import numpy as np
import scipy as sp
from scipy import ndimage
import os, sys, getopt
from ctypes import *
import bagOfns as bg
import time

def ERA(exit, Pmod, Psup):
    out = Pmod(exit)
    out = Psup(out)
    return out

def HIO(exit, Pmod, Psup, beta=1.):
    out = Pmod(exit)
    out = exit + beta * Psup( (1+1/beta)*out - 1/beta * exit ) - beta * out  
    return out

def Thibault(exit, Pmod, Psup):
    out = Psup(exit)
    out = exit +  Pmod(2*out - exit) - out
    return out

def makeExits3(sample, probe, coords):
    """Calculate the exit surface waves with no wrapping and assuming integer coordinate shifts"""
    exits = np.zeros((len(coords), probe.shape[0], probe.shape[1]), dtype=np.complex128)
    for i, coord in enumerate(coords):
        exits[i] = sample[-coord[0]:probe.shape[0]-coord[0], -coord[1]:probe.shape[1]-coord[1]]
    exits *= probe 
    return exits


def makeExits2(sample, probe, coords, exits):
    """Calculate the exit surface waves with no wrapping and assuming integer coordinate shifts"""
    for i, coord in enumerate(coords):
        exits[i] = sample[-coord[0]:probe.shape[0]-coord[0], -coord[1]:probe.shape[1]-coord[1]]
    exits *= probe 
    return exits


def makeExits(sample, probe, coords):
    """Calculate the exit surface waves with possible wrapping using the Fourier shift theorem"""
    exits = np.zeros((len(coords), probe.shape[0], probe.shape[1]), dtype=np.complex128)
    for i in range(len(coords)):
        exits[i] = bg.roll(sample, coords[i])[:probe.shape[0], :probe.shape[1]]
    exits *= probe 
    return exits

    
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
