import numpy as np
import scipy as sp
from scipy import ndimage
import os, sys, getopt
from ctypes import *
import bagOfns as bg
import time

#------------------------------------------------------
# coordinates for 1d coords and 1d sample
#------------------------------------------------------
def fmod(prob, coords):
    exits      = makeExits_1dsample(prob.sample, prob.probe, coords)
    diffAmps   = np.abs(bg.fftn(exits)) * prob.mask
    return np.sum((diffAmps - prob.diffAmps)**2) 

def Pmod_hat_diffs(diffAmps, psis, mask = None, alpha = 1.0e-10):
    if mask == None :
        mask = np.ones_like(diffAmps[0], dtype=np.bool)
    exits_out = psis
    exits_out = exits_out * (mask * diffAmps / np.clip(np.abs(exits_out), alpha, np.inf) \
                   + (~mask) )
    return exits_out

def emod_grad_coords_1d(T, probe, coords, diffAmps):
    exits_d = bg.fftn(makeExits_grad_1d(T, probe, coords, coords_d = None))
    exits   = bg.fftn(makeExits_1dsample(T, probe, coords))
    exits   = np.conj(exits_d) * (exits - Pmod_hat_diffs(diffAmps, exits))
    out     = np.sum(np.real(exits), axis = 2)
    return 2.0 * np.sum(out, axis = 1)

def emod_grad_dot_coords_1d(coords_d, T, probe, coords, diffAmps):
    exits_d = bg.fftn(makeExits_grad_1d(T, probe, coords, coords_d))
    exits   = bg.fftn(makeExits_1dsample(T, probe, coords))
    exits   = np.conj(exits_d) * (exits - Pmod_hat_diffs(diffAmps, exits))
    return 2 * np.sum(np.real(exits) )

def sample_grad_trans_1d(T, coord, coord_d):
    """Calculate : F-1[ Ti_hat (-2 pi i / Nx) Ri_d n ]"""
    Nx       = T.shape[-1]
    x        = bg.make_xy([Nx], origin=(0,0))
    #
    array    = np.zeros_like(T)
    array[:] = -2.0J * np.pi * (coord_d[1] * x / float(Nx)) * np.exp(-2.0J * np.pi * (coord[1] * x / float(Nx)))
    return bg.ifftn_1d(bg.fftn_1d(T) * array)

def makeExits_grad_1d_old(T, probe, coords, coords_d):
    """Calculate the exit surface waves but with T_i = sample_grad_trans_1d(T, coord, coord_d)
    
    if coords_d == None then T_i = sample_grad_trans_1d(T, coord, I)"""
    exits = np.zeros((len(coords), probe.shape[0], probe.shape[1]), dtype=np.complex128)
    #
    if np.all(coords_d == None):
        I = np.ones_like(coords[0])
        s_calc = False
    else :
        s_calc = True
    #
    for i in range(len(coords)):
        if s_calc :
            T_g      = sample_grad_trans_1d(T, coords[i], coords_d[i])
        else :
            T_g      = sample_grad_trans_1d(T, coords[i], I)
        exits[i] = T_g[:probe.shape[0], :probe.shape[1]]
    exits *= probe 
    return exits

def makeExits_grad_1d(sample, probe, coords, coords_d):
    sample1d        = sample[0, :]
    sample_stack    = np.zeros((len(coords), sample1d.shape[0]), dtype=sample1d.dtype)
    sample_stack[:] = sample1d
    # 
    # make the phase ramp for each 1d sample in the stack
    x_stack = np.zeros_like(sample_stack)
    x       = bg.make_xy([sample1d.shape[0]], origin=(0,))
    x       = -2.0J * np.pi * (x / float(sample1d.shape[0]))
    if np.all(coords_d == None ):
        for i, coord in enumerate(coords):
            x_stack[i] = np.exp(x * coords[i][1])
    else :
        for i, coord in enumerate(coords):
            x_stack[i] = np.exp(x * coords[i][1]) * coords_d[i][1]  
    x_stack = x_stack * x 
    #
    # shift the whole sample stack
    sample_stack  = bg.fftn_1d(sample_stack)
    sample_stack  = sample_stack * x_stack
    sample_stack  = bg.ifftn_1d(sample_stack)
    sample_stack  = sample_stack[:, :probe.shape[1]]
    #
    # make the exit waves
    exits = np.zeros((len(coords), probe.shape[0], probe.shape[1]), dtype=np.complex128)
    for i in range(len(coords)):
        exits[i, :] = sample_stack[i]  
    return exits * probe

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

def makeExits_1dsample(sample, probe, coords):
    sample1d        = sample[0, :]
    sample_stack    = np.zeros((len(coords), sample1d.shape[0]), dtype=sample1d.dtype)
    sample_stack[:] = sample1d
    # 
    # make the phase ramp for each 1d sample in the stack
    x_stack = np.zeros_like(sample_stack)
    x       = bg.make_xy([sample1d.shape[0]], origin=(0,))
    x       = -2.0J * np.pi * (x / float(sample1d.shape[0]))
    for i, coord in enumerate(coords):
        x_stack[i] = x * coord[1]
    #
    # shift the whole sample stack
    sample_stack  = bg.fftn_1d(sample_stack)
    sample_stack  = sample_stack * np.exp(x_stack)
    sample_stack  = bg.ifftn_1d(sample_stack)
    sample_stack  = sample_stack[:, :probe.shape[1]]
    #
    # make the exit waves
    exits = np.zeros((len(coords), probe.shape[0], probe.shape[1]), dtype=np.complex128)
    for i in range(len(coords)):
        exits[i, :] = sample_stack[i]  
    return exits * probe
