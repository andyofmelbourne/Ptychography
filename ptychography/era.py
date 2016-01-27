import numpy as np
import sys
from itertools import product

from era_mpi import ERA_mpi
from era_gpu import ERA_gpu

def ERA(I, R, P, O, iters, OP_iters = 1, mask = 1, background = None, method = None, hardware = 'cpu', alpha = 1.0e-10, dtype=None, full_output = True):
    """
    Find the phases of 'I' given O, P, R using the Error Reduction Algorithm (Ptychography).
    
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
    
    iters : int
        The number of ERA iterations to perform.

    OP_iters : int, optional, default (1)
        The number of projections onto the sample and probe consistency constraints 
        before doing the modulus projection when update = 'OP' has been selected. 
        Ideally OP_iters should be large enough so that convergence has been acheived.

    mask : numpy.ndarray, (M, K), optional, default (1)
        The valid detector pixels. Mask[i, j] = 1 (or True) when the detector pixel 
        i, j is valid, Mask[i, j] = 0 otherwise.
    
    method : (None, 1, 2, 3, 4), optional, default (None)
        method = None :
            Automattically choose method 1, 2 or 3 based on the contents of 'O' and 'P'.
            if   O == None and P == None then method = 3
            elif O == None then method = 1
            elif P == None then method = 2
        method = 1 :
            Just update 'O'
        method = 2 :
            Just update 'P'
        method = 3 :
            Update 'O' and 'P'
        method = 4 :
            Update 'O' and 'background'
        method = 5 :
            Update 'P' and 'background'
        method = 6 :
            Update 'O', 'P' and 'background'
    
    hardware : ('cpu', 'gpu', 'mpi'), optional, default ('cpu') 
        Choose to run the reconstruction on a single cpu core ('cpu'), a single gpu
        ('gpu') or many cpu's ('mpi'). The numerical results should be identical.
    
    alpha : float, optional, default (1.0e-10)
        A floating point number to regularise array division (prevents 1/0 errors).
    
    dtype : (None, 'single' or 'double'), optional, default ('single')
        Determines the numerical precision of the calculation. If dtype==None, then
        it is determined from the datatype of I.

    full_output : bool, optional, default (True)
        If true then return a bunch of diagnostics (see info) as a python dictionary 
        (a list of key : value pairs).

    
    Returns
    -------
    O / P : numpy.ndarray, (U, V) / (M, K)
        If update = 'O': returns the transmission function of the real space object, 
        retrieved after 'iters' iterations of the ERA algorithm. If update = 'P': 
        returns the probe wavefront. If update = 'OP' then you get both (O, P).
    
    info : dict, optional
        contains diagnostics:
            
            'exits' : the exit surface waves corresponding to the returned O (P)
            'I'     : the diffraction patterns corresponding to 'exits' above
            'eMod'  : the modulus error for each iteration:
                      eMod_i = sqrt( sum(| exits - Pmod(exits) |^2) / I )
            'eCon'  : the convergence error for each iteration:
                      eCon_i = sqrt( sum(| O_i - O_i-1 |^2) / sum(| O_i |^2) )
        

    Notes 
    -----
    The ERA is the simplest iterative projection algorithm. It proceeds by 
    progressive projections of the exit surface waves onto the set of function that 
    satisfy the:
        modulus constraint : after propagation to the detector the exit surface waves
                             must have the same modulus (square root of the intensity) 
                             as the detected diffraction patterns (the I's).
        
        consistency constraint : the exit surface waves (W) must be separable into some object 
                                 and probe functions so that W_n = O_n x P.
    
    The 'projection' operation onto one of these constraints makes the smallest change to the set 
    of exit surface waves (in the Euclidean sense) that is required to satisfy said constraint.

    Examples 
    --------
    """
    if hardware == 'gpu':
        return ERA_gpu(I, R, P, O, iters, OP_iters, mask, background, method, hardware, alpha, dtype, full_output)
    elif hardware == 'mpi':
        return ERA_mpi(I, R, P, O, iters, OP_iters, mask, background, method, hardware, alpha, dtype, full_output)

    if method == None :
        if O is None and P is None :
            method = 3
        elif O is None :
            method = 1
        elif P is None :
            method = 2

        if background is not None :
            method += 3
    
    if method == 1 or method == 4 : 
        update = 'O'
    elif method == 2 or method == 5 : 
        update = 'P'
    elif method == 3 or method == 6 : 
        update = 'OP'
    
    if dtype is None :
        dtype   = I.dtype
        c_dtype = (I[0,0,0] + 1J * I[0, 0, 0]).dtype
    
    elif dtype == 'single':
        dtype   = np.float32
        c_dtype = np.complex64

    elif dtype == 'double':
        dtype   = np.float64
        c_dtype = np.complex128

    if O is None :
        # find the smallest array that fits O
        # This is just U = M + R[:, 0].max() - R[:, 0].min()
        #              V = K + R[:, 1].max() - R[:, 1].min()
        shape = (I.shape[1] + R[:, 0].max() - R[:, 0].min(),\
                 I.shape[2] + R[:, 1].max() - R[:, 1].min())
        O = np.ones(shape, dtype = c_dtype)
    
    if P is None :
        P = np.random.random(I[0].shape) + 1J*np.random.random(I[0].shape)
    
    P = P.astype(c_dtype)
    O = O.astype(c_dtype)
    
    I_norm    = np.sum(mask * I)
    amp       = np.sqrt(I).astype(dtype)
    exits     = make_exits(O, P, R)
    P_heatmap = None
    O_heatmap = None
    eMods     = []
    eCons     = []

    # subtract an overall offset from R's
    R[:, 0] -= R[:, 0].max()
    R[:, 1] -= R[:, 1].max()
    
    # method 1 and 2, update O or P
    #---------
    if method == 1 or method == 2 :
        print 'algrithm progress iteration convergence modulus error'
        for i in range(iters) :
            if update == 'O': bak = O.copy()
            if update == 'P': bak = P.copy()
            E_bak        = exits.copy()
            
            # modulus projection 
            exits        = pmod_1(amp, exits, mask, alpha = alpha)
            
            E_bak       -= exits

            # consistency projection 
            if update == 'O': O, P_heatmap = psup_O_1(exits, P, R, O.shape, P_heatmap, alpha = alpha)
            if update == 'P': P, O_heatmap = psup_P_1(exits, O, R, O_heatmap, alpha = alpha)
            
            exits = make_exits(O, P, R, exits)
            
            # metrics
            if update == 'O': temp = O
            if update == 'P': temp = P
            
            bak -= temp
            eCon   = np.sum( (bak * bak.conj()).real ) / np.sum( (temp * temp.conj()).real )
            eCon   = np.sqrt(eCon)
            
            eMod   = np.sum( (E_bak * E_bak.conj()).real ) / I_norm
            eMod   = np.sqrt(eMod)
            
            update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )

            eMods.append(eMod)
            eCons.append(eCon)
        
        if full_output : 
            info = {}
            info['exits'] = exits
            info['I']     = np.abs(np.fft.fftn(exits, axes = (-2, -1)))**2
            info['eMod']  = eMods
            info['eCon']  = eCons
            info['heatmap']  = P_heatmap
            if update == 'O': return O, info
            if update == 'P': return P, info
        else :
            if update == 'O': return O
            if update == 'P': return P

    # method 3
    #---------
    elif method == 3 : 
        print 'algrithm progress iteration convergence modulus error'
        for i in range(iters) :
            OP_bak = np.hstack((O.ravel().copy(), P.ravel().copy()))
            E_bak  = exits.copy()
            
            # modulus projection 
            exits        = pmod_1(amp, exits, mask, alpha = alpha)
            
            E_bak       -= exits
            
            # consistency projection 
            for j in range(OP_iters):
                O, P_heatmap = psup_O_1(exits, P, R, O.shape, None, alpha = alpha)
                P, O_heatmap = psup_P_1(exits, O, R, None, alpha = alpha)
            
            exits = make_exits(O, P, R, exits)
            
            # metrics
            temp = np.hstack((O.ravel(), P.ravel()))
            
            OP_bak-= temp
            eCon   = np.sum( (OP_bak * OP_bak.conj()).real ) / np.sum( (temp * temp.conj()).real )
            eCon   = np.sqrt(eCon)
            
            eMod   = np.sum( (E_bak * E_bak.conj()).real ) / I_norm
            eMod   = np.sqrt(eMod)
            
            update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )

            eMods.append(eMod)
            eCons.append(eCon)
        
        if full_output : 
            info = {}
            info['exits'] = exits
            info['I']     = np.abs(np.fft.fftn(exits, axes = (-2, -1)))**2
            info['eMod']  = eMods
            info['eCon']  = eCons
            info['heatmap']  = P_heatmap
            return O, P, info
        else :
            return O, P

    # method 4 or 5
    #---------
    # update the object with background retrieval
    elif method == 4 or method == 5 :
        if background is None :
            background = np.random.random((I.shape)).astype(dtype)
        else :
            temp       = np.empty(I.shape, dtype = dtype)
            temp[:]    = np.sqrt(background)
            background = temp
        
        print 'algrithm progress iteration convergence modulus error'
        for i in range(iters) :
            if update == 'O': bak = O.copy()
            if update == 'P': bak = P.copy()
            E_bak        = exits.copy()
            
            # modulus projection 
            exits, background  = pmod_7(amp, background, exits, mask, alpha = alpha)
            
            E_bak       -= exits

            # consistency projection 
            if update == 'O': O, P_heatmap = psup_O_1(exits, P, R, O.shape, P_heatmap, alpha = alpha)
            if update == 'P': P, O_heatmap = psup_P_1(exits, O, R, O_heatmap, alpha = alpha)

            background[:] = np.mean(background, axis=0)
            
            exits = make_exits(O, P, R, exits)
            
            # metrics
            if update == 'O': temp = O
            if update == 'P': temp = P
            
            bak   -= temp
            eCon   = np.sum( (bak * bak.conj()).real ) / np.sum( (temp * temp.conj()).real )
            eCon   = np.sqrt(eCon)
            
            eMod   = np.sum( (E_bak * E_bak.conj()).real ) / I_norm
            eMod   = np.sqrt(eMod)
            
            update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )

            eMods.append(eMod)
            eCons.append(eCon)
        
        if full_output : 
            info = {}
            info['exits'] = exits
            info['I']     = np.abs(np.fft.fftn(exits, axes = (-2, -1)))**2
            info['eMod']  = eMods
            info['eCon']  = eCons
            info['heatmap']  = P_heatmap
            if update == 'O': return O, background**2, info
            if update == 'P': return P, background**2, info
        else :
            if update == 'O': return O, background**2
            if update == 'P': return P, background**2

    elif method == 6 : 
        if background is None :
            background = np.random.random((I.shape)).astype(dtype)
        else :
            temp       = np.empty(I.shape, dtype = dtype)
            temp[:]    = np.sqrt(background)
            background = temp
        
        print 'algrithm progress iteration convergence modulus error'
        for i in range(iters) :
            OP_bak = np.hstack((O.ravel().copy(), P.ravel().copy()))
            E_bak  = exits.copy()
            
            # modulus projection 
            exits, background  = pmod_7(amp, background, exits, mask, alpha = alpha)
            
            E_bak       -= exits
            
            # consistency projection 
            for j in range(OP_iters):
                O, P_heatmap = psup_O_1(exits, P, R, O.shape, None, alpha = alpha)
                P, O_heatmap = psup_P_1(exits, O, R, None, alpha = alpha)
            
            background[:] = np.mean(background, axis=0)
            
            exits = make_exits(O, P, R, exits)
            
            # metrics
            temp = np.hstack((O.ravel(), P.ravel()))
            
            OP_bak-= temp
            eCon   = np.sum( (OP_bak * OP_bak.conj()).real ) / np.sum( (temp * temp.conj()).real )
            eCon   = np.sqrt(eCon)
            
            eMod   = np.sum( (E_bak * E_bak.conj()).real ) / I_norm
            eMod   = np.sqrt(eMod)
            
            update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )

            eMods.append(eMod)
            eCons.append(eCon)
        
        if full_output : 
            info = {}
            info['exits'] = exits
            info['I']     = np.abs(np.fft.fftn(exits, axes = (-2, -1)))**2
            info['eMod']  = eMods
            info['eCon']  = eCons
            info['heatmap']  = P_heatmap
            return O, P, background**2, info
        else :
            return O, P, background**2

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

def make_exits(O, P, R, exits = None):
    if exits is None :
        exits = np.empty((len(R),) + P.shape, dtype = P.dtype)
    
    for i, r in enumerate(R) : 
        exits[i] = multiroll(O, [r[0], r[1]])[:P.shape[0], :P.shape[1]] * P
    return exits

def psup_O_1(exits, P, R, O_shape, P_heatmap = None, alpha = 1.0e-10):
    O = np.zeros(O_shape, P.dtype)
    
    # Calculate denominator
    #----------------------
    # but only do this if it hasn't been done already
    # (we must set P_heatmap = None when the probe/coords has changed)
    if P_heatmap is None : 
        P_heatmap = make_P_heatmap(P, R, O_shape)

    # Calculate numerator
    #--------------------
    for r, exit in zip(R, exits):
        #print exit.shape, P.shape, O.shape, -r[0],P.shape[0]-r[0], -r[1],P.shape[1]-r[1], O[-r[0]:P.shape[0]-r[0], -r[1]:P.shape[1]-r[1]].shape
        O[-r[0]:P.shape[0]-r[0], -r[1]:P.shape[1]-r[1]] += exit * P.conj()
         
    # divide
    #-------
    O  = O / (P_heatmap + alpha)
    
    return O, P_heatmap

def psup_P_1(exits, O, R, O_heatmap = None, alpha = 1.0e-10):
    P = np.zeros(exits[0].shape, exits.dtype)
    
    # Calculate denominator
    #----------------------
    # but only do this if it hasn't been done already
    # (we must set O_heatmap = None when the object/coords has changed)
    if O_heatmap is None : 
        O_heatmap = make_O_heatmap(O, R, P.shape)

    # Calculate numerator
    #--------------------
    Oc = O.conj()
    for r, exit in zip(R, exits):
        P += exit * Oc[-r[0]:P.shape[0]-r[0], -r[1]:P.shape[1]-r[1]] 
         
    # divide
    #-------
    P  = P / (O_heatmap + alpha)
    
    return P, O_heatmap

def make_P_heatmap(P, R, shape):
    P_heatmap = np.zeros(shape, dtype = P.real.dtype)
    P_temp    = np.zeros(shape, dtype = P.real.dtype)
    P_temp[:P.shape[0], :P.shape[1]] = (P.conj() * P).real
    for r in R : 
        P_heatmap += multiroll(P_temp, [-r[0], -r[1]]) 
    return P_heatmap
    
def make_O_heatmap(O, R, shape):
    O_heatmap = np.zeros(O.shape, dtype = O.real.dtype)
    O_temp    = (O * O.conj()).real
    for r in R : 
        O_heatmap += multiroll(O_temp, [r[0], r[1]]) 
    return O_heatmap[:shape[0], :shape[1]]

def pmod_1(amp, exits, mask = 1, alpha = 1.0e-10):
    exits = np.fft.fftn(exits, axes = (-2, -1))
    exits = Pmod_1(amp, exits, mask = mask, alpha = alpha)
    exits = np.fft.ifftn(exits, axes = (-2, -1))
    return exits
    
def Pmod_1(amp, exits, mask = 1, alpha = 1.0e-10):
    exits  = mask * exits * amp / (abs(exits) + alpha)
    exits += (1 - mask) * exits
    return exits

def pmod_7(amp, background, exits, mask = 1, alpha = 1.0e-10):
    exits = np.fft.fftn(exits, axes = (-2, -1))
    exits, background = Pmod_7(amp, background, exits, mask = mask, alpha = alpha)
    exits = np.fft.ifftn(exits, axes = (-2, -1))
    return exits, background
    
def Pmod_7(amp, background, exits, mask = 1, alpha = 1.0e-10):
    M = mask * amp / np.sqrt((exits.conj() * exits).real + background**2 + alpha)
    exits      *= M
    background *= M
    exits += (1 - mask) * exits
    return exits, background

def multiroll(x, shift, axis=None):
    """Roll an array along each axis.

    Thanks to: Warren Weckesser, 
    http://stackoverflow.com/questions/30639656/numpy-roll-in-several-dimensions
    
    
    Parameters
    ----------
    x : array_like
        Array to be rolled.
    shift : sequence of int
        Number of indices by which to shift each axis.
    axis : sequence of int, optional
        The axes to be rolled.  If not given, all axes is assumed, and
        len(shift) must equal the number of dimensions of x.

    Returns
    -------
    y : numpy array, with the same type and size as x
        The rolled array.

    Notes
    -----
    The length of x along each axis must be positive.  The function
    does not handle arrays that have axes with length 0.

    See Also
    --------
    numpy.roll

    Example
    -------
    Here's a two-dimensional array:

    >>> x = np.arange(20).reshape(4,5)
    >>> x 
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])

    Roll the first axis one step and the second axis three steps:

    >>> multiroll(x, [1, 3])
    array([[17, 18, 19, 15, 16],
           [ 2,  3,  4,  0,  1],
           [ 7,  8,  9,  5,  6],
           [12, 13, 14, 10, 11]])

    That's equivalent to:

    >>> np.roll(np.roll(x, 1, axis=0), 3, axis=1)
    array([[17, 18, 19, 15, 16],
           [ 2,  3,  4,  0,  1],
           [ 7,  8,  9,  5,  6],
           [12, 13, 14, 10, 11]])

    Not all the axes must be rolled.  The following uses
    the `axis` argument to roll just the second axis:

    >>> multiroll(x, [2], axis=[1])
    array([[ 3,  4,  0,  1,  2],
           [ 8,  9,  5,  6,  7],
           [13, 14, 10, 11, 12],
           [18, 19, 15, 16, 17]])

    which is equivalent to:

    >>> np.roll(x, 2, axis=1)
    array([[ 3,  4,  0,  1,  2],
           [ 8,  9,  5,  6,  7],
           [13, 14, 10, 11, 12],
           [18, 19, 15, 16, 17]])

    """
    x = np.asarray(x)
    if axis is None:
        if len(shift) != x.ndim:
            raise ValueError("The array has %d axes, but len(shift) is only "
                             "%d. When 'axis' is not given, a shift must be "
                             "provided for all axes." % (x.ndim, len(shift)))
        axis = range(x.ndim)
    else:
        # axis does not have to contain all the axes.  Here we append the
        # missing axes to axis, and for each missing axis, append 0 to shift.
        missing_axes = set(range(x.ndim)) - set(axis)
        num_missing = len(missing_axes)
        axis = tuple(axis) + tuple(missing_axes)
        shift = tuple(shift) + (0,)*num_missing

    # Use mod to convert all shifts to be values between 0 and the length
    # of the corresponding axis.
    shift = [s % x.shape[ax] for s, ax in zip(shift, axis)]

    # Reorder the values in shift to correspond to axes 0, 1, ..., x.ndim-1.
    shift = np.take(shift, np.argsort(axis))

    # Create the output array, and copy the shifted blocks from x to y.
    y = np.empty_like(x)
    src_slices = [(slice(n-shft, n), slice(0, n-shft))
                  for shft, n in zip(shift, x.shape)]
    dst_slices = [(slice(0, shft), slice(shft, n))
                  for shft, n in zip(shift, x.shape)]
    src_blks = product(*src_slices)
    dst_blks = product(*dst_slices)
    for src_blk, dst_blk in zip(src_blks, dst_blks):
        y[dst_blk] = x[src_blk]

    return y
