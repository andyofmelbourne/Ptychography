import numpy as np
import era 

def Back_projection(I, R, P, O, mask = 1, alpha = 1.0e-10, full_output = True):
    """
    Back project the diffraction images by assuming that the phases of the diffraction
    pattern are that of the probe.
    
    Parameters
    ----------
    I : numpy.ndarray, (N, M, K)
        Diffraction patterns to be phased. 
    
        N : the number of diffraction patterns
        M : the number of pixels along slow scan axis of the detector
        K : the number of pixels along fast scan axis of the detector
    
    R : numpy.ndarray, (N, 3)
        The translational displacements of the object corresponding to each diffraction
        pattern in pixel units, such that:
            O_n = O[i - R[n, 0], j - R[n, 1]]
        This way positive numbers move the sample to the left, or, we can think of the 
        coordinates (R) as tracking the poisition of some point in the object.
    
    P : numpy.ndarray, (M, K)
        The wavefront of the real space probe incident on the surface of
        the detector.
    
    O : numpy.ndarray, (U, V) 
        The transmission function of the object (multiplicative) so that:
            I_n = |F[ O_n x P ]|^2
        where I_n and O_n are the n'th diffraction pattern and object respectively and
        F[.] is the 2D Fourier transform of '.'. The field of view of the object may be
        infinite but no smaller than the probe so that:
            M <= U < inf
            K <= V < inf
    
    mask : numpy.ndarray, (M, K), optional, default (1)
        The valid detector pixels. Mask[i, j] = 1 (or True) when the detector pixel 
        i, j is valid, Mask[i, j] = 0 otherwise.

    alpha : float, optional, default (1.0e-10)
        A floating point number to regularise array division (prevents 1/0 errors).
    
    full_output : bool, optional, default (True)
        If true then return a bunch of diagnostics (see info) as a python dictionary 
        (a list of key : value pairs).
    
    Returns
    -------
    O : numpy.ndarray, (U, V) / (M, K)
        Returns the transmission function of the real space object, 
    
    info : dict, optional
        contains diagnostics:
            
            'exits'   : the exit surface waves corresponding to the returned O (P)
            'heatmap' : the integrated intensity of the probe over the sample surface
    """

    if O is None :
        # find the smallest array that fits O
        # This is just U = M + R[:, 0].max() - R[:, 0].min()
        #              V = K + R[:, 1].max() - R[:, 1].min()
        shape = (I.shape[1] + R[:, 0].max() - R[:, 0].min(),\
                 I.shape[2] + R[:, 1].max() - R[:, 1].min())
        O = np.ones(shape, dtype = P.dtype)
    
    # subtract an overall offset from R's
    R[:, 0] -= R[:, 0].max()
    R[:, 1] -= R[:, 1].max()
    print R

    amp   = np.sqrt(I.astype(np.float64))
    exits = np.zeros(I.shape, dtype = np.complex64)
    P_F   = np.fft.fftn(P)
    exits = mask * amp * np.exp(1J * np.angle(P_F))
    exits = np.fft.ifftn(exits, axes = (-2, -1))

    O, P_heatmap = era.psup_O_1(exits, P, R, O.shape, None, alpha = alpha)

    if full_output :
        info = {}
        info['exits'] = exits
        info['heatmap'] = P_heatmap
        return O, info
    else :
        return O
