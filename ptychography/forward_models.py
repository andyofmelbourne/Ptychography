"""
Forward simulate Ptychographic problems.
"""

import numpy as np
import bagOfns as bg
from numpy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq
from era import make_exits 


def forward_sim(shape_P = (32, 64), shape_O = (128, 128), A = 32, defocus = 100.,\
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

    defocus : float, optional, default (100.)
        The distance between the focus of the probe and the sample in 
        pixel units. Note that dx / wavelength has been defined as unity.
        Where dx is the pixel width in the plane of the sample.

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

    M : numpy.ndarray, bool, (N, M)
        The masked detector pixels, is False if the pixel is bad or 'masked'
        and True otherwise.

    P : numpy.ndarray, complex128, (N, M)
        The real-space probe in the plane of the sample. The probe is centred.

    O : numpy.ndarray, complex128, (U, V)
        The real-space sample transmission function.

    B : numpy.ndarray, float64 or int64, (N, M)
        The constant (different for every pixel but the same for every diffraction
        pattern) background. Has the same dtype as I.

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
    i     = fftfreq(shape[0], 1/float(shape[0]))
    j     = fftfreq(shape[1], 1/float(shape[1]))
    i, j  = np.meshgrid(i, j, indexing='ij')
    P     = fftshift(i**2 + j**2) < A
    P     = P.astype(np.complex128) * photons_pupil
    
    # defocus
    # set dx / wavelength = 1, now defocus is in 'pixels'
    qy, qx  = fftfreq(shape[0]), fftfreq(shape[1])
    qy, qx  = np.meshgrid(qy, qx, indexing='ij')
    qy, qx  = fftshift(qy), fftshift(qx)
    q2      = qy**2 + qx**2
    exp     = np.exp(-1.0J * np.pi * defocus * q2)
    
    P *= exp
    P  = fftshift(ifftn(ifftshift(P)))

    # Sample (O)
    #-------
    amp     = bg.scale(bg.brog(shape_O), 0.0, 1.0)
    phase   = bg.scale(bg.twain(shape_O), -np.pi, np.pi)
    O       = amp * np.exp(1J * phase)

    # Sample coords (R)
    #--------------
    #ys, xs = np.arange(P.shape[0] - O.shape[0], 1, ny), np.arange(P.shape[1] - O.shape[1], 1, nx)
    ys, xs = np.rint(np.linspace(1, P.shape[0] - O.shape[0], ny)), np.rint(np.linspace(1, P.shape[1] - O.shape[1], nx))
    ys, xs = np.meshgrid(ys.astype(np.int), xs.astype(np.int), indexing = 'ij')
    R = np.array(zip(ys.ravel(), xs.ravel()))

    # random offset 
    if random_offset is not None :
        dcoords = (np.random.random(R.shape) * int(random_offset)).astype(np.int)
        R += dcoords
        R[np.where(R > 0)] = 0
        R[:, 0][np.where(R[:, 0] < shape_illum[0] - shape_O[0])] = shape_illum[0] - shape_O[0]
        R[:, 1][np.where(R[:, 1] < shape_illum[1] - shape_O[1])] = shape_illum[1] - shape_O[1]

    # Diffraction patterns (I)
    #-------------------------
    exits = np.zeros((len(R),) + P.shape, dtype=np.complex128)
    exits = make_exits(O, P, R, exits)
    exits = fftn(exits, axes = (-2, -1))
    I     = (exits.conj() * exits).real
    
    # Background (B)
    if background is not None :
        B  = np.random.random(I[0].shape) * background
        B  = np.rint(B)
        I += B
    
    # Detector mask (M)
    if mask is not None :
        M_ss = (np.random.random(mask) * I.shape[1]).astype(np.int)
        M_fs = (np.random.random(mask) * I.shape[2]).astype(np.int)
        M    = np.ones(I[0].shape, dtype=np.bool)
        M[M_ss, M_fs] = False

    # Counting Statistics
    if counting_statistics :
        I = np.random.poisson(lam = I)
    
    return I, R, mask, P, O, B.astype(I.dtype)







    
