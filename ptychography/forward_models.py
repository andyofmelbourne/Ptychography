# -*- coding: utf-8 -*-
"""
Forward simulate Ptychographic problems.
"""

import numpy as np
import bagOfns as bg
#from numpy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq
#from numpy.fft import fftfreq
from era import make_exits, pmod_1

def forward_sim(shape_P = (32, 64), shape_O = (128, 128), A = 14, defocus = 0.,\
                photons_pupil = 1, ny = 10, nx = 10, random_offset = None, \
                background = None, mask = None, counting_statistics = False):
    """
    Make a 'vanilla' Ptychographic example.

    Parameters
    ----------
    shape_P : tuple, (N, M), optional, default (32, 64)
        array dimensions of the probe function and thus the detector 
        dimensions also. Where:
        N : the number of pixels along slow scan axis of the detector
        M : the number of pixels along fast scan axis of the detector

    shape_O : tuple, (U, V), optional, default (128, 128)
        array dimensions of the sample transmission function. Where:
        
        U : the number of pixels for the sample parallel to the 
            slow scan axis of the detector
        V : the number of pixels for the sample parallel to the 
            fast scan axis of the detector

    A : integer, optional, default (32)
        The radius of the circular pupil function (pixels) at the detector.

    defocus : float, optional, default (10.)
        The distance between the focus of the probe and the sample in 
        inverse Fresnel numbers. So:
            defocus = 1/F = wavelength dz / X^2 
                          = du^2 dz / wavelength z^2 
        where X is the field of view of the Probe, du is the pixel
        size (assumed square), z is the detector distance and dz is 
        the defocus distance in meters. If defocus is 0 then F is 
        infinite and Probe is not propagated, while if the defocus 
        is ~ 1 (F ~ 1) then we are in the far-field and the Fresnel
        propagation is undersampled (your probe will be crap). 
        
        Setting defocus to a complex number tells the routine to 
        use a Fourier Fresnel propapgator, so that:
            defocus = (focus-sample distance) + i(sample-detector distance)
        where (again) the distances are in inverse Fresnel numbers.
        The real-space probe is then given by:
            P = F-1{ A_n exp(-i pi defocus_r n^2) }
        and the exit wave at the detector by:
            exit = F{ P O exp(i pi n^2 / N^2 defocus_i) }
        
    photons_pupil : scalar, optional, default (1)
        The average number of photons per pixel in the pupil function.
        When counting_statistics is True then this parameter effectively
        sets the poisson noise level.­Otherwise this simply rescales the
        diffraction intensities.
    
    ny : integer, optional, default (10)
        The number of sample scan points parallel to the slow scan 
        axis of the detector for each fast scan position.

    nx : integer, optional, default (10)
        The number of sample scan points parallel to the fast scan 
        axis of the detector for each slow scan position.

    random_offset : None or scalar, optional, default (None)
        If not None then a random transverse displacement is added to 
        each sample position, ranging from 0 to 'random_offset' pixels.

    background : None or scalar, optional, default (None)
        If not None then a random number between 0 and 'background'
        is added to each pixel. The sample background is applied to
        each diffraction pattern.

    mask : None or integer, default (None)
        If not None then 'mask' pixels are randomly chosen and added
        to the mask (M). The same pixels are masked for each diffraction
        pattern. Note that nothing is actually done to the masked pixels
        in the diffraction patterns.

    counting_statistics : bool, optional, default (False)  
        If True then poisson counting statistics is added for each pixel
        in each diffraction pattern. The mean of the poisson distribution
        is roughly proportional to 'photons_pupil' (also by the particular
        sample transmission function).
    
    Returns
    -------
    return I, R, M, P, O, B

    I : numpy.ndarray, float64 or int64, (K, N, M)
        The diffraction patterns, K is the total number of scan positions,
        equal to ny*nx. If counting_statistics is True then I has the dtype
        numpy.int64 otherwise it is numpy.float64.
    
    R : numpy.ndarray, int64, (K, 2)
        The fast and slow scan postions of the sample corresponding to each 
        of the K diffraction measurements. R[:, 0] are the positions of the 
        sample parallel to the slow scan axis of the detector and R[:, 1] 
        to the fast scan axis of the detector.

    M : numpy.ndarray or 1, bool, (N, M)
        The masked detector pixels, is False if the pixel is bad or 'masked'
        and True otherwise. If 'mask' is None then M is 1. 

    P : numpy.ndarray, complex128, (N, M)
        The real-space probe in the plane of the sample. The probe is centred.

    O : numpy.ndarray, complex128, (U, V)
        The real-space sample transmission function.

    B : numpy.ndarray or 0, float64 or int64, (N, M)
        The constant (different for every pixel but the same for every diffraction
        pattern) background. Has the same dtype as I. If 'background' is None 
        the B is 0.

    Notes 
    -----
    The probe is convergent with a circular farfield pupil.
    The sample is Brog and twain. 
    """
    # detector shape
    shape = shape_P

    # Probe (P)
    #------
    # far-field circular pupil
    i     = np.fft.fftfreq(shape[0], 1/float(shape[0]))
    j     = np.fft.fftfreq(shape[1], 1/float(shape[1]))
    i, j  = np.meshgrid(i, j, indexing='ij')
    P     = i**2 + j**2 < A**2
    P     = P.astype(np.complex128) * np.sqrt(photons_pupil)
    
    # defocus
    # set wavelength . z / detector pixel size = 1, and set wavelength = 1
    # now defocus is in units of (wavelength . z / detector pixel size)
    q2      = i**2 + j**2
    exp     = np.exp(-1.0J * np.pi * defocus.real * q2)
    
    P *= exp
    P = np.fft.fftshift(np.fft.ifftn(P))

    # If defocus has an imaginary component then add
    # the Fresnel phase factor
    if np.abs(defocus.imag) > 0.0 :
        # apply phase
        exps = np.exp(1.0J * np.pi * (i**2 / (defocus.imag * P.shape[0]**2) + \
                                      j**2 / (defocus.imag * P.shape[1]**2)))
        def prop(x):
            out = np.fft.ifftn(np.fft.ifftshift(x, axes=(-2,-1)), axes = (-2, -1)) * exps.conj() 
            out = np.fft.fftn(out, axes = (-2, -1))
            return out
        
        def iprop(x):
            out = np.fft.ifftn(x, axes = (-2, -1)) * exps
            out = np.fft.fftn(out, axes = (-2, -1))
            return np.fft.ifftshift(out, axes=(-2,-1))
        
        P = iprop(np.fft.fftn(np.fft.ifftshift(P)))
    else :
        def prop(x):
            out = np.fft.ifftshift(x, axes = (-2,-1))
            out = np.fft.fftn(out, axes = (-2, -1))
            return out
        
        def iprop(x):
            out = np.fft.ifftn(x, axes = (-2, -1))
            return np.fft.fftshift(out, axes = (-2, -1))
    
    # Sample (O)
    #-------
    amp     = bg.scale(bg.lena(shape_O), 0.0, 1.0)
    phase   = bg.scale(bg.twain(shape_O), -0.1*np.pi, 0.1*np.pi)
    O       = amp * np.exp(1J * phase)

    # Sample coords (R)
    #--------------
    #ys, xs = np.arange(P.shape[0] - O.shape[0], 1, ny), np.arange(P.shape[1] - O.shape[1], 1, nx)
    ys, xs = np.rint(np.linspace(0, P.shape[0] - O.shape[0] , ny)), \
             np.rint(np.linspace(0, P.shape[1] - O.shape[1] , nx))
    ys, xs = np.meshgrid(ys.astype(np.int), xs.astype(np.int), indexing = 'ij')
    R = np.array(zip(ys.ravel(), xs.ravel()))

    # random offset 
    if random_offset is not None :
        dcoords = (np.random.random(R.shape) * int(random_offset)).astype(np.int)
        R += dcoords
        R[np.where(R > 0)] = 0
        R[:, 0][np.where(R[:, 0] < shape_P[0] - shape_O[0] )] = shape_P[0] - shape_O[0]
        R[:, 1][np.where(R[:, 1] < shape_P[1] - shape_O[1] )] = shape_P[1] - shape_O[1] 

    R[:, 0] -= R[:, 0].max()
    R[:, 1] -= R[:, 1].max()
    
    # Diffraction patterns (I)
    #-------------------------
    exits = np.zeros((len(R),) + P.shape, dtype=np.complex128)
    exits = make_exits(O, P, R, exits)
    exits = prop(exits)
    I     = (exits.conj() * exits).real
    I     = np.fft.ifftshift(I, axes=(-2, -1))
    
    # Background (B)
    if background is not None :
        B  = np.random.random(I[0].shape) * background
        B  = np.rint(B)
        I += B
    else :
        B = 0
    
    # Detector mask (M)
    if mask is not None :
        M_ss = (np.random.random(mask) * I.shape[1]).astype(np.int)
        M_fs = (np.random.random(mask) * I.shape[2]).astype(np.int)
        M    = np.ones(I[0].shape, dtype=np.bool)
        M[M_ss, M_fs] = False
    else :
        M = 1

    # Counting Statistics
    if counting_statistics :
        I = np.random.poisson(lam = I)
    
    if B is not 0 :
        B = B.astype(I.dtype)
    
    # calculate the error
    #amp   = ifftshift(np.sqrt(I), axes=(-2, -1))
    #exits = ifftn(exits, axes = (-2, -1))
    #exits, eMod = pmod_1(amp, exits, M, alpha = 1.0e-10, eMod_calc = True)
    #print np.sqrt(eMod / np.sum(I))

    return I, R, M, P, O, B, prop, iprop




if __name__ == '__main__':
    print '\nMaking the forward simulated data...'
    I, R, M, P, O, B = forward_sim(shape_P = (128, 128), shape_O = (256, 256), A = 32, defocus = 0.0e-2 + 10.0J,\
				      photons_pupil = 1, ny = 10, nx = 10, random_offset = 5, \
				      background = None, mask = 100, counting_statistics = False)
