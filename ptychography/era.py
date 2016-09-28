import numpy as np
import sys

#import era

from itertools import product
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def ERA(I, R, P, O, iters, OP_iters = 1, mask = 1, Fresnel = False, background = None, method = None, Pmod_probe = False, probe_centering = False, hardware = 'cpu', alpha = 1.0e-10, dtype=None, sample_blur = None, full_output = True, verbose = False, output_h5file = None, output_h5group = None, output_h5interval = 1):
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

    OP_iters : int or tuple, optional, default (1)
        The number of projections onto the sample and probe consistency constraints 
        before doing the modulus projection when update = 'OP' has been selected. 
        If OP_iters is a tuple (say) (5, 10) then the probe and object will only be
        updated every 10 iterations, for the other 9 iterations the object alone
        is updated. Ideally OP_iters should be large enough so that convergence has 
        been acheived.

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

    Pmod_probe : Flase or int, optional, default (False)
        If Pmod_probe == int then the modulus of the far-field probe is enforced
        after every update of the probe for the first 'Pmod_probe' iterations. To
        always enforce the far-field modulus then set Pmod_probe = np.inf.
    
    hardware : ('cpu', 'gpu', 'mpi'), optional, default ('cpu') 
        Choose to run the reconstruction on a single cpu core ('cpu'), a single gpu
        ('gpu') or many cpu's ('mpi'). The numerical results should be identical. 
        The gpu routines are currently experimental so they are disabled by default.
    
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
    if rank == 0 : print '\n\nERA_mpi v5'    

    method, update, dtype, c_dtype, MPI_dtype, MPI_c_dtype, OP_iters, O, P, amp, background, R, mask, I_norm, N, exits, Fresnel = \
            preamble(I, R, P, O, iters, OP_iters, mask, background, method, hardware, alpha, dtype, Fresnel, full_output)

    prop, iprop = make_prop(Fresnel, P.shape)
    Pamp = prop(P)
    Pamp = np.sqrt((Pamp.conj() * Pamp).real)
    
    P_heatmap = None
    O_heatmap = None
    eMods     = []
    eCons     = []
    
    # method 1, 2 or 3, update O, P or OP 
    #---------
    if method == 1 or method == 2 or method == 3 :
        if rank == 0 : print 'algrithm progress iteration convergence modulus error'
        for i in range(iters) :
            if rank == 0 : 
                if update == 'O' : bak = O.copy()
                if update == 'P' : bak = P.copy()
                if update == 'OP': bak = np.hstack((O.ravel().copy(), P.ravel().copy()))
            
            
            # modulus projection 
            exits, eMod = pmod_1(amp, exits, mask, alpha = alpha, eMod_calc = True, prop = (prop, iprop))
            
            # consistency projection 
            if update == 'O': O, P_heatmap = psup_O(exits, P, R, O.shape, P_heatmap, alpha, MPI_dtype, MPI_c_dtype, verbose, sample_blur)
            if update == 'P': P, O_heatmap = psup_P(exits, O, R, O_heatmap, alpha, MPI_dtype, MPI_c_dtype)
            if update == 'OP':
                if i % OP_iters[1] == 0 :
                    for j in range(OP_iters[0]):
                        O, P_heatmap = psup_O(exits, P, R, O.shape, None, alpha, MPI_dtype, MPI_c_dtype, verbose, sample_blur)
                        P, O_heatmap = psup_P(exits, O, R, None, alpha, MPI_dtype, MPI_c_dtype)

                    # only centre if both P and O are updated
                    if probe_centering :
                        # get the centre of mass of |P|^2
                        import scipy.ndimage
                        a  = (P.conj() * P).real
                        cm = np.rint(scipy.ndimage.measurements.center_of_mass(a)).astype(np.int) - np.array(a.shape)/2
                        
                        # centre P
                        P = multiroll(P, -cm)
                        
                        # shift O
                        O = multiroll(O, -cm)
                        
                        P_heatmap = O_heatmap = None
                        
                else :
                    O, P_heatmap = psup_O(exits, P, R, O.shape, P_heatmap, alpha, MPI_dtype, MPI_c_dtype, verbose, sample_blur)
            
            # enforce the modulus of the far-field probe 
            if Pmod_probe is not None and i < Pmod_probe :
                P = Pmod_P(Pamp, P, mask, alpha, prop = (prop, iprop))

            exits = make_exits(O, P, R, exits)
            
            # metrics
            eMod   = comm.reduce(eMod, op=MPI.SUM)

            if rank == 0 :
                if update == 'O' : temp = O
                if update == 'P' : temp = P
                if update == 'OP': temp = np.hstack((O.ravel(), P.ravel()))
                 
                bak -= temp
                eCon   = np.sum( (bak * bak.conj()).real ) / np.sum( (temp * temp.conj()).real )
                eCon   = np.sqrt(eCon)
                
                eMod   = np.sqrt(eMod / I_norm)
                
                update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )
                
                eMods.append(eMod)
                eCons.append(eCon)
                
                # output O, P and eMod 
                ######################
                if (output_h5file is not None) and (i % output_h5interval == 0) :
                    import h5py
                    f = h5py.File(output_h5file)
                    key = output_h5group + '/O'
                    if key in f :
                        del f[key]
                    f[key] = O
                    
                    key = output_h5group + '/P'
                    if key in f :
                        del f[key]
                    f[key] = P
                    
                    key = output_h5group + '/eMod'
                    if key in f :
                        del f[key]
                    f[key] = np.array(eMods)
                    f.close()
        
    # method 4 or 5 or 6
    #---------
    # update the object with background retrieval
    elif method == 4 or method == 5 or method == 6 :
        
        if rank == 0 : print 'algrithm progress iteration convergence modulus error'
        for i in range(iters) :
            if rank == 0 : 
                if update == 'O' : bak = O.copy()
                if update == 'P' : bak = P.copy()
                if update == 'OP': bak = np.hstack((O.ravel().copy(), P.ravel().copy()))
            
            # modulus projection 
            exits, background, eMod = pmod_7(amp, background, exits, mask, alpha = alpha, eMod_calc = True)
            
            # consistency projection 
            if update == 'O': O, P_heatmap = psup_O(exits, P, R, O.shape, P_heatmap, alpha, MPI_dtype, MPI_c_dtype, verbose, sample_blur)
            if update == 'P': P, O_heatmap = psup_P(exits, O, R, O_heatmap, alpha, MPI_dtype, MPI_c_dtype, sample_blur)
            if update == 'OP':
                if i % OP_iters[1] == 0 :
                    for j in range(OP_iters[0]):
                        O, P_heatmap = psup_O(exits, P, R, O.shape, None, alpha, MPI_dtype, MPI_c_dtype, verbose, sample_blur)
                        P, O_heatmap = psup_P(exits, O, R, None, alpha, MPI_dtype, MPI_c_dtype)
                    
                    # only centre if both P and O are updated
                    if probe_centering :
                        # get the centre of mass of |P|^2
                        import scipy.ndimage
                        a  = (P.conj() * P).real
                        cm = np.rint(scipy.ndimage.measurements.center_of_mass(a)).astype(np.int) - np.array(a.shape)/2
                        
                        if rank == 0 : print 'probe cm:', cm
                        
                        # centre P
                        P = multiroll(P, -cm)
                        
                        # shift O
                        O = multiroll(O, -cm)
                        
                        P_heatmap = O_heatmap = None
                        
                else :
                    O, P_heatmap = psup_O(exits, P, R, O.shape, P_heatmap, alpha, MPI_dtype, MPI_c_dtype, verbose, sample_blur)
            
            # enforce the modulus of the far-field probe 
            if Pmod_probe is not None and i < Pmod_probe :
                P = Pmod_P(Pamp, P, mask, alpha, prop = (prop, iprop))
            
            backgroundT  = np.mean(background, axis=0)
            backgroundTT = np.empty_like(backgroundT)
            comm.Allreduce([backgroundT, MPI_dtype], \
                           [backgroundTT,  MPI_dtype], \
                           op=MPI.SUM)
            background[:] = backgroundTT / float(size)
            
            exits = make_exits(O, P, R, exits)
            
            # metrics
            eMod   = comm.allreduce(eMod, op=MPI.SUM)

            if rank == 0 :
                if update == 'O': temp = O
                if update == 'P': temp = P
                if update == 'OP': temp = np.hstack((O.ravel(), P.ravel()))
                 
                bak -= temp
                eCon   = np.sum( (bak * bak.conj()).real ) / np.sum( (temp * temp.conj()).real )
                eCon   = np.sqrt(eCon)
                
                eMod   = np.sqrt(eMod / I_norm)
                
                update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )
                
                eMods.append(eMod)
                eCons.append(eCon)
        
    # This should not be necessary but it crashes otherwise
    #I = np.fft.fftshift(np.abs(np.fft.fftn(exits, axes = (-2, -1)))**2, axes = (-2, -1))
    per_diff_eMod = model_error_per_diff(amp, exits, mask, background = 0, prop = prop)
       
    I = np.fft.fftshift(np.abs(prop(exits))**2, axes = (-2,-1))
    if rank == 0 :
        I_rec = [I.copy()]
        for i in range(1, size):
            #print 'gathering I from rank:', i
            I_rec.append( comm.recv(source = i, tag = i) )
        I = np.array([e for es in I_rec for e in es])
    else :
        comm.send(I, dest=0, tag=rank)

    if rank == 0 :
        info = {}
        info['I']       = I
        info['eMod']    = eMods

        # combine the per diff eMods
        err = per_diff_eMod[0]
        for i in range(1, size):
            err = np.hstack((err, per_diff_eMod[i]))
        info['eMod_diff'] = err
        info['eCon']    = eCons
        info['heatmap'] = P_heatmap
        if background is not None :
            if len(background.shape) == 3 :
                background = background[0]
            info['background'] = np.fft.fftshift(background)**2
        return O, P, info
    else :
        return None, None, None


def make_prop(Fresnel, shape):
    if Fresnel is not False :
        i     = np.fft.fftfreq(shape[0], 1/float(shape[0]))
        j     = np.fft.fftfreq(shape[1], 1/float(shape[1]))
        i, j  = np.meshgrid(i, j, indexing='ij')
        
        # apply phase
        exps = np.exp(1.0J * np.pi * (i**2 * Fresnel / shape[0]**2 + \
                                      j**2 * Fresnel / shape[1]**2))
        def prop(x):
            out = np.fft.ifftn(np.fft.ifftshift(x, axes=(-2,-1)), axes = (-2, -1)) * exps.conj() 
            out = np.fft.fftn(out, axes = (-2, -1))
            return out
        
        def iprop(x):
            out = np.fft.ifftn(x, axes = (-2, -1)) * exps
            out = np.fft.fftn(out, axes = (-2, -1))
            return np.fft.ifftshift(out, axes=(-2,-1))

        #P = iprop(np.fft.fftn(np.fft.ifftshift(P)))
    else :
        def prop(x):
            out = np.fft.ifftshift(x, axes = (-2,-1))
            out = np.fft.fftn(out, axes = (-2, -1))
            return out
        
        def iprop(x):
            out = np.fft.ifftn(x, axes = (-2, -1))
            return np.fft.fftshift(out, axes = (-2, -1))
    return prop, iprop

def model_error_per_diff(amp, exits, mask, background = 0, prop = None):
    if prop is None :
        exits = np.fft.fftn(exits, axes = (-2, -1))
    else :
        exits = prop(exits)
    M     = np.sqrt((exits.conj() * exits).real + background**2)
    err   = np.sqrt(np.sum( mask * (M - amp)**2, axis=(1,2) ) /  np.sum( mask * amp**2, axis=(1,2) ))
    err_tot = comm.gather(err, root=0)
    return err_tot


def psup_O(exits, P, R, O_shape, P_heatmap = None, alpha = 1.0e-10, MPI_dtype = MPI.DOUBLE, MPI_c_dtype = MPI.DOUBLE_COMPLEX, verbose = False, sample_blur = None):
    OT = np.zeros(O_shape, P.dtype)
    
    # Calculate denominator
    #----------------------
    # but only do this if it hasn't been done already
    # (we must set P_heatmap = None when the probe/coords has changed)
    if P_heatmap is None : 
        P_heatmapT = make_P_heatmap(P, R, O_shape)
        P_heatmap  = np.empty_like(P_heatmapT)
        #comm.Allreduce([P_heatmapT, MPI.__TypeDict__[P_heatmapT.dtype.char]], \
        #               [P_heatmap,  MPI.__TypeDict__[P_heatmap.dtype.char]], \
        #               op=MPI.SUM)
        comm.Allreduce([P_heatmapT, MPI_dtype], \
                       [P_heatmap,  MPI_dtype], \
                       op=MPI.SUM)

    # Calculate numerator
    #--------------------
    for r, exit in zip(R, exits):
        OT[-r[0]:P.shape[0]-r[0], -r[1]:P.shape[1]-r[1]] += exit * P.conj()
    
    # divide
    # here we need to do an all reduce
    #---------------------------------
    O = np.empty_like(OT)
    #comm.Allreduce([OT, MPI.__TypeDict__[OT.dtype.char]], \
    #               [O, MPI.__TypeDict__[O.dtype.char]],   \
    #                op=MPI.SUM)
    comm.Allreduce([OT, MPI_c_dtype], \
                   [O, MPI_c_dtype],  \
                    op=MPI.SUM)
    comm.Barrier()
    O  = O / (P_heatmap + alpha)

    if sample_blur is not None :
        import scipy.ndimage
        O.real = scipy.ndimage.gaussian_filter(O.real, sample_blur, mode='wrap')
        O.imag = scipy.ndimage.gaussian_filter(O.imag, sample_blur, mode='wrap')
    
    # set a maximum value for the amplitude of the object
    O = np.clip(np.abs(O), 0.0, 2.0) * np.exp(1.0J * np.angle(O))
    return O, P_heatmap

def psup_P(exits, O, R, O_heatmap = None, alpha = 1.0e-10, MPI_dtype = MPI.DOUBLE, MPI_c_dtype = MPI.DOUBLE_COMPLEX):
    PT = np.zeros(exits[0].shape, exits.dtype)
    
    # Calculate denominator
    #----------------------
    # but only do this if it hasn't been done already
    # (we must set O_heatmap = None when the object/coords has changed)
    if O_heatmap is None : 
        O_heatmapT = np.ascontiguousarray(make_O_heatmap(O, R, PT.shape))
        #O_heatmapT = era.make_O_heatmap(O, R, PT.shape) produces a non-contig. array for some reason
        O_heatmap  = np.empty_like(O_heatmapT)
        comm.Allreduce([O_heatmapT, MPI_dtype], \
                       [O_heatmap,  MPI_dtype], \
                       op=MPI.SUM)

    # Calculate numerator
    #--------------------
    Oc = O.conj()
    for r, exit in zip(R, exits):
        PT += exit * Oc[-r[0]:PT.shape[0]-r[0], -r[1]:PT.shape[1]-r[1]] 
         
    # divide
    #-------
    P = np.empty_like(PT)
    comm.Allreduce([PT, MPI_c_dtype], \
                   [P, MPI_c_dtype],   \
                    op=MPI.SUM)
    P  = P / (O_heatmap + alpha)
    
    return P, O_heatmap

def make_P_heatmap(P, R, shape):
    P_heatmap = np.zeros(shape, dtype = P.real.dtype)
    #P_temp    = np.zeros(shape, dtype = P.real.dtype)
    #P_temp[:P.shape[0], :P.shape[1]] = (P.conj() * P).real
    P_temp = (P.conj() * P).real
    for r in R : 
        #P_heatmap += multiroll(P_temp, [-r[0], -r[1]]) 
        P_heatmap[-r[0]:P.shape[0]-r[0], -r[1]:P.shape[1]-r[1]] += P_temp
    return P_heatmap

def make_O_heatmap(O, R, shape):
    O_heatmap = np.zeros(O.shape, dtype = O.real.dtype)
    O_temp    = (O * O.conj()).real
    for r in R : 
        O_heatmap += multiroll(O_temp, [r[0], r[1]]) 
    return O_heatmap[:shape[0], :shape[1]]

def chunkIt(seq, num):
    splits = np.mgrid[0:len(seq):(num+1)*1J].astype(np.int)
    out    = []
    for i in range(splits.shape[0]-1):
        out.append(seq[splits[i]:splits[i+1]])
    return out

def make_exits(O, P, R, exits = None):
    if exits is None :
        exits = np.empty((len(R),) + P.shape, dtype = P.dtype)
    
    for i, r in enumerate(R) : 
        #print O.shape, r[0], r[1], P.shape, exits.shape
        exits[i] = multiroll(O, [r[0], r[1]])[:P.shape[0], :P.shape[1]] * P
    return exits

def preamble(I, R, P, O, iters, OP_iters, mask, background, method, hardware, alpha, dtype, Fresnel, full_output):
    """
    This routine takes all of the arguements of ERA and applies all the boring tasks to the input before 
    a reconstruction.
    """
    if method == None :
        if O is None and P is None :
            method = 3
        elif O is None :
            method = 1
        elif P is None :
            method = 2

        if background is not None :
            method += 3
    method  = comm.bcast(method, root=0)
    
    if method == 1 or method == 4 : 
        update = 'O'
    elif method == 2 or method == 5 : 
        update = 'P'
    elif method == 3 or method == 6 : 
        update = 'OP'

    if type(OP_iters) == int :
        OP_iters = (OP_iters, 1)
    
    if rank == 0 :
        if dtype is None :
            if I.dtype == np.float32 :
                dtype = 'single'
            else :
                dtype = 'double'
        
        if dtype == 'single':
            dtype       = np.float32
            MPI_dtype   = MPI.FLOAT
            c_dtype     = np.complex64
            MPI_c_dtype = MPI.COMPLEX

        elif dtype == 'double':
            dtype       = np.float64
            MPI_dtype   = MPI.DOUBLE
            c_dtype     = np.complex128
            MPI_c_dtype = MPI.DOUBLE_COMPLEX

        if O is None :
            # find the smallest array that fits O
            # This is just U = M + R[:, 0].max() - R[:, 0].min()
            #              V = K + R[:, 1].max() - R[:, 1].min()
            shape = (I.shape[1] + R[:, 0].max() - R[:, 0].min(),\
                     I.shape[2] + R[:, 1].max() - R[:, 1].min())
            O = np.ones(shape, dtype = c_dtype)
        
        if P is None :
            print 'initialising the probe with random numbers...'
            P = np.random.random(I[0].shape) + 1J*np.random.random(I[0].shape)
        
        P = P.astype(c_dtype)
        O = O.astype(c_dtype)
        
        I_norm    = np.sum(mask * I)
        N         = len(I)
        amp       = np.sqrt(I).astype(dtype)
        amp       = np.fft.ifftshift(amp, axes=(-2, -1))
        mask      = np.fft.ifftshift(mask)

        #P = np.fft.ifft( np.fft.ifftshift( np.fft.fftn(P) ) )

        # subtract an overall offset from R's
        R[:, 0] -= R[:, 0].max()
        R[:, 1] -= R[:, 1].max()

    else :
        amp = dtype = c_dtype = I_norm = N = None

    # now we need to share the info with everyone
    dtype   = comm.bcast(dtype, root=0)
    c_dtype = comm.bcast(c_dtype, root=0)
    O       = comm.bcast(O, root=0)
    P       = comm.bcast(P, root=0)
    mask    = comm.bcast(mask, root=0)
    N       = comm.bcast(N, root=0)
    Fresnel = comm.bcast(Fresnel, root=0)
    
    # for some reason these don't like to be bcast?
    if dtype == np.float32:
        MPI_dtype   = MPI.FLOAT
        MPI_c_dtype = MPI.COMPLEX
    else :
        MPI_dtype   = MPI.DOUBLE
        MPI_c_dtype = MPI.DOUBLE_COMPLEX
    
    # split the coords 
    if rank == 0 :
        R = chunkIt(R, size)
    R = comm.scatter(R, root=0)

    # split the diffraction ampiltudes
    if rank == 0 :
        amp = chunkIt(amp, size)
    amp = comm.scatter(amp, root=0)

    exits = make_exits(O, P, R)
    
    # background
    if background is None and method in [4,5,6]:
        background = np.random.random((exits.shape)).astype(dtype) + 0.1
    elif method in [4,5,6]:
        temp       = np.empty(exits.shape, dtype = dtype)
        temp[:]    = np.sqrt(np.fft.ifftshift(background))
        background = temp

    return method, update, dtype, c_dtype, MPI_dtype, MPI_c_dtype, OP_iters, O, P, amp, background, R, mask, I_norm, N, exits, Fresnel

def pmod_1(amp, exits, mask = 1, alpha = 1.0e-10, eMod_calc = False, prop = None):
    if prop is None :
        exits = np.fft.fftn(exits, axes = (-2, -1))
    else :
        exits = prop[0](exits)
    
    exits, eMod = Pmod_1(amp, exits, mask, alpha, eMod_calc)
    
    if prop is None :
        exits = np.fft.ifftn(exits, axes = (-2, -1))
    else :
        exits = prop[1](exits)
    
    if eMod_calc : 
        return exits, eMod
    else :
        return exits
    
def Pmod_1(amp, exits, mask = 1, alpha = 1.0e-10, eMod_calc = False):
    M = np.sqrt((exits.conj() * exits).real) + alpha
    if eMod_calc :
        eMod = np.sum((M - amp)**2 * mask)
    else :
        eMod = None
    M = amp / M
    if mask is not 1 :
        i = np.where(mask == 0)
        if len(i[0]) > 0 :
            M[:, i[0], i[1]] = 1.
    exits *= M
    return exits, eMod

def pmod_7(amp, background, exits, mask = 1, alpha = 1.0e-10, eMod_calc = False):
    exits = np.fft.fftn(exits, axes = (-2, -1))
    exits, background, eMod = Pmod_7(amp, background, exits, mask, alpha, eMod_calc)
    exits = np.fft.ifftn(exits, axes = (-2, -1))
    if eMod_calc :
        return exits, background, eMod
    else :
        return exits, background
    
def Pmod_7(amp, background, exits, mask = 1, alpha = 1.0e-10, eMod_calc = False):
    M = np.sqrt((exits.conj() * exits).real + background**2 + alpha)
    if eMod_calc :
        eMod = np.sum((M - amp)**2 * mask)
    else :
        eMod = None
    M = mask * amp / M
    if mask is 1 :
        i = np.where(1-mask)
        M[:, i[0], i[1]] = 1.
    exits      *= M
    background *= M
    return exits, background, eMod

def Pmod_P(amp, P, mask = 1, alpha = 1.0e-10, prop = None):
    if prop is None :
        P = np.fft.fftn(P, axes = (-2, -1))
    else :
        P = prop[0](P)
    
    M = np.sqrt((P.conj() * P).real) + alpha
    M = amp / M

    if mask is not 1 :
        i = np.where(mask == 0)
        if len(i[0]) > 0 :
            M[i[0], i[1]] = 1.
    P *= M
    
    if prop is None :
        P = np.fft.ifftn(P, axes = (-2, -1))
    else :
        P = prop[1](P)
    return P

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

if __name__ == '__main__' :
    import numpy as np
    import time
    import sys

    from era import ERA
    from dm import DM
    from back_projection import Back_projection
    from forward_models import forward_sim
    from display import write_cxi
    from display import display_cxi

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) == 2 :
        iters = int(sys.argv[1])
        test = 'all'
    elif len(sys.argv) == 3 :
        iters = int(sys.argv[1])
        test = sys.argv[2]
    else :
        iters = 10
        test = 'all'

    # Many cpu cores 
    #----------------
    if test in ['1', '2', '3', 'all']:
        if rank == 0 :
            print '\nMaking the forward simulated data...'
            I, R, M, P, O, B = forward_sim(shape_P = (128, 128), shape_O = (256, 256), A = 32, defocus = 1.0e-2,\
                                              photons_pupil = 1, ny = 10, nx = 10, random_offset = None, \
                                              background = None, mask = 100, counting_statistics = False)
            # make the masked pixels bad
            I += 10000. * ~M 
            
            # initial guess for the probe 
            P0 = np.fft.fftshift( np.fft.ifftn( np.abs(np.fft.fftn(P)) ) )
        else :
            I = R = O = P = P0 = M = B = None
    
    if test == 'all' or test == '1':
        if rank == 0 : 
            print '\n-------------------------------------------'
            print 'Updating the object on a single cpu core...'
            d0 = time.time()

        Or, info = ERA_mpi(I, R, P, None, iters, mask=M, method = 1, hardware = 'mpi', alpha=1e-10, dtype='double')
        
        if rank == 0 : 
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
            
            write_cxi(I, info['I'], P, P, O, Or, \
                      R, None, None, None, M, info['eMod'], fnam = 'output_method1.cxi')

    if test == 'all' or test == '2':
        if rank == 0 : 
            print '\n-------------------------------------------'
            print '\nUpdating the probe on a single cpu core...'
            d0 = time.time()

        Pr, info = ERA_mpi(I, R, P0, O, iters, mask=M, method = 2, hardware = 'mpi', alpha=1e-10)
        
        if rank == 0 : 
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
            
            write_cxi(I, info['I'], P, Pr, O, O, \
                      R, None, None, None, M, info['eMod'], fnam = 'output_method2.cxi')

    if test == 'all' or test == '3':
        if rank == 0 : 
            print '\n-------------------------------------------'
            print '\nUpdating the object and probe on a single cpu core...'
            d0 = time.time()

        Or, Pr, info = ERA_mpi(I, R, P0, None, iters, mask=M, method = 3, hardware = 'mpi', alpha=1e-10)

        if rank == 0 : 
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
            
            write_cxi(I, info['I'], P, Pr, O, Or, \
                      R, None, None, None, M, info['eMod'], fnam = 'output_method3.cxi')
    

    if test in ['4', '5', '6', 'all']:
        if rank == 0 :
            print '\n\n\nMaking the forward simulated data with background...'
            I, R, M, P, O, B = forward_sim(shape_P = (128, 128), shape_O = (256, 256), A = 32, defocus = 1.0e-2,\
                                              photons_pupil = 100, ny = 10, nx = 10, random_offset = None, \
                                              background = 10, mask = 100, counting_statistics = False)
            # make the masked pixels bad
            I += 10000. * ~M 
            
            # initial guess for the probe 
            P0 = np.fft.fftshift( np.fft.ifftn( np.abs(np.fft.fftn(P)) ) )
        else :
            I = R = O = P = P0 = M = B = None
    
    if test == 'all' or test == '4':
        if rank == 0 : 
            print '\n-------------------------------------------'
            print 'Updating the object and background on a single cpu core...'
            d0 = time.time()

        Or, info = ERA_mpi(I, R, P, None, iters, mask=M, method = 4, hardware = 'mpi', alpha=1e-10, dtype='double')

        if rank == 0 : 
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
            
            write_cxi(I, info['I'], P, P, O, Or, \
                      R, None, B, info['background'], M, info['eMod'], fnam = 'output_method4.cxi')

    if test == 'all' or test == '5':
        if rank == 0 : 
            print '\n-------------------------------------------'
            print '\nUpdating the probe and background on a single cpu core...'
            d0 = time.time()

        Pr, info = ERA_mpi(I, R, P0, O, iters, mask=M, method = 5, hardware = 'mpi', alpha=1e-10)

        if rank == 0 : 
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
            
            write_cxi(I, info['I'], P, Pr, O, O, \
                      R, None, B, info['background'], M, info['eMod'], fnam = 'output_method5.cxi')

    if test == 'all' or test == '6':
        if rank == 0 : 
            print '\n-------------------------------------------'
            print '\nUpdating the object and probe and background on a single cpu core...'
            d0 = time.time()

        Or, Pr, info = ERA_mpi(I, R, P0, None, iters, mask=M, method = 6, hardware = 'mpi', alpha=1e-10)

        if rank == 0 : 
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
            
            write_cxi(I, info['I'], P, Pr, O, Or, \
                      R, None, B, info['background'], M, info['eMod'], fnam = 'output_method6.cxi')


