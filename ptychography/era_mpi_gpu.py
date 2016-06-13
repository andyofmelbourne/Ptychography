import numpy as np
import sys

import era

from mpi4py import MPI

from era_gpu import pmod_4
from era_gpu import Pmod_4
from era_gpu import pmod_back
from era_gpu import Pmod_back

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def ERA_mpi_gpu(I, R, P, O, iters, OP_iters = 1, mask = 1, background = None, method = None, hardware = 'cpu', alpha = 1.0e-10, dtype=None, full_output = True):
    """
    MPI variant of ptychography.ERA
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
    
    if method == 1 or method == 4 : 
        update = 'O'
    elif method == 2 or method == 5 : 
        update = 'P'
    elif method == 3 or method == 6 : 
        update = 'OP'
    
    if rank == 0 :
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

        # subtract an overall offset from R's
        R[:, 0] -= R[:, 0].max()
        R[:, 1] -= R[:, 1].max()

    else :
        amp = dtype = c_dtype = None

    P_heatmap = None
    O_heatmap = None
    eMods     = []
    eCons     = []
    
    # now we need to share the info with everyone
    dtype   = comm.bcast(dtype, root=0)
    c_dtype = comm.bcast(c_dtype, root=0)
    O       = comm.bcast(O, root=0)
    P       = comm.bcast(P, root=0)
    mask    = comm.bcast(mask, root=0)

    # split the coords 
    if rank == 0 :
        R = chunkIt(R, size)
    R = comm.scatter(R, root=0)

    # split the diffraction ampiltudes
    if rank == 0 :
        amp = chunkIt(amp, size)
    amp = comm.scatter(amp, root=0)

    # make our exit waves
    exits     = era.make_exits(O, P, R)

    # set up the gpu
    #---------------
    #---------------
    import pyfft
    import pyopencl
    import pyopencl.array
    from   pyfft.cl import Plan
    import pyopencl.clmath 

    # get the CUDA platform
    if rank == 0 : print 'opencl platforms found:'
    platforms = pyopencl.get_platforms()
    for p in platforms:
        if rank == 0 : print '\t', p.name
        if p.name == 'NVIDIA CUDA':
            platform = p
            if rank == 0 : print '\tChoosing', p.name
    
    # get one of the gpu's device id
    if rank == 0 : print '\nopencl devices found:'
    devices = platform.get_devices()
    for d in devices:
        if rank == 0 : print '\t', d.name

    if rank % 2 == 0 :
        if rank == 0 : print '\t', rank ,'Choosing', devices[0].name
        device = devices[0]
    else :
        device = devices[1]

    
    # create a context for the device
    context = pyopencl.Context([device])
    
    # create a command queue for the device
    queue = pyopencl.CommandQueue(context)
    
    # make a plan for the ffts
    plan = Plan(amp[0].shape, dtype=c_dtype, queue=queue)

    exits_g = pyopencl.array.to_device(queue, np.ascontiguousarray(exits))
    amp_g   = pyopencl.array.to_device(queue, np.ascontiguousarray(amp))
    if mask is not 1 :
        mask_g    = np.empty(amp.shape, dtype=np.int8)
        mask_g[:] = mask.astype(np.int8)*2 - 1
        mask_g    = pyopencl.array.to_device(queue, np.ascontiguousarray(mask_g))
    else :
        mask_g  = 1
    
    # method 1, 2 or 3, update O, P or OP 
    #---------
    if method == 1 or method == 2 or method == 3 :
        if rank == 0 : print 'algrithm progress iteration convergence modulus error'
        for i in range(iters) :
            if rank == 0 : 
                if update == 'O' : bak = O.copy()
                if update == 'P' : bak = P.copy()
                if update == 'OP': bak = np.hstack((O.ravel().copy(), P.ravel().copy()))
            
            E_bak        = exits.copy()
            
            # modulus projection 
            exits_g.set(exits)
            exits        = pmod_4(amp_g, exits_g, plan, mask_g, alpha = alpha).get()
            
            E_bak       -= exits

            # consistency projection 
            if update == 'O': O, P_heatmap = psup_O(exits, P, R, O.shape, P_heatmap, alpha = alpha)
            if update == 'P': P, O_heatmap = psup_P(exits, O, R, O_heatmap, alpha = alpha)
            if update == 'OP':
                for j in range(OP_iters):
                    O, P_heatmap = psup_O(exits, P, R, O.shape, None, alpha = alpha)
                    P, O_heatmap = psup_P(exits, O, R, None, alpha = alpha)
            
            exits = era.make_exits(O, P, R, exits)
            
            # metrics
            eMod   = np.sum( (E_bak * E_bak.conj()).real ) 
            eMod   = comm.allreduce(eMod, op=MPI.SUM)

            if rank == 0 :
                if update == 'O' : temp = O
                if update == 'P' : temp = P
                if update == 'OP': temp = np.hstack((O.ravel(), P.ravel()))
                 
                bak -= temp
                eCon   = np.sum( (bak * bak.conj()).real ) / np.sum( (temp * temp.conj()).real )
                eCon   = np.sqrt(eCon)
                
                eMod   = np.sqrt(eMod / I_norm)
                
                era.update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )
                
                eMods.append(eMod)
                eCons.append(eCon)
        
        if full_output : 
            if rank == 0 :
                info = {}
                info['exits'] = exits
                info['I']     = np.abs(np.fft.fftn(exits, axes = (-2, -1)))**2
                info['eMod']  = eMods
                info['eCon']  = eCons
                info['heatmap']  = P_heatmap
                if update == 'O' : return O, info
                if update == 'P' : return P, info
                if update == 'OP': return O, P, info
            else :
                if update == 'OP': 
                    return None, None, None
                else :
                    return None, None
        else :
            if rank == 0 :
                if update == 'O' : return O
                if update == 'P' : return P
                if update == 'OP': return O, P
            else :
                if update == 'OP': 
                    return None, None
                else :
                    return None

    # method 4 or 5 or 6
    #---------
    # update the object with background retrieval
    elif method == 4 or method == 5 or method == 6 :
        if background is None :
            background = np.random.random((exits.shape)).astype(dtype)
        else :
            temp       = np.empty(exits.shape, dtype = dtype)
            temp[:]    = np.sqrt(background)
            background = temp
        
        background_g   = pyopencl.array.to_device(queue, np.ascontiguousarray(background))
        
        if rank == 0 : print 'algrithm progress iteration convergence modulus error'
        for i in range(iters) :
            if rank == 0 : 
                if update == 'O' : bak = O.copy()
                if update == 'P' : bak = P.copy()
                if update == 'OP': bak = np.hstack((O.ravel().copy(), P.ravel().copy()))
            
            E_bak        = exits.copy()
            
            # modulus projection 
            exits_g.set(exits)
            background_g.set(background)
            exits_g, background_g = pmod_back(amp_g, background_g, exits_g, plan, mask_g, alpha = alpha)
            exits      = exits_g.get()
            background = background_g.get()
            
            E_bak       -= exits
            
            # consistency projection 
            if update == 'O': O, P_heatmap = psup_O(exits, P, R, O.shape, P_heatmap, alpha = alpha)
            if update == 'P': P, O_heatmap = psup_P(exits, O, R, O_heatmap, alpha = alpha)
            if update == 'OP':
                for j in range(OP_iters):
                    O, P_heatmap = psup_O(exits, P, R, O.shape, None, alpha = alpha)
                    P, O_heatmap = psup_P(exits, O, R, None, alpha = alpha)

            #background[:] = np.mean(background, axis=0)
            backgroundT  = np.mean(background, axis=0)
            backgroundTT = np.empty_like(backgroundT)
            comm.Allreduce([backgroundT, MPI.__TypeDict__[backgroundT.dtype.char]], \
                           [backgroundTT,  MPI.__TypeDict__[backgroundTT.dtype.char]], \
                           op=MPI.SUM)
            background[:] = backgroundTT / float(size)
            
            exits = era.make_exits(O, P, R, exits)
            
            # metrics
            eMod   = np.sum( (E_bak * E_bak.conj()).real ) 
            eMod   = comm.allreduce(eMod, op=MPI.SUM)

            if rank == 0 :
                if update == 'O': temp = O
                if update == 'P': temp = P
                if update == 'OP': temp = np.hstack((O.ravel(), P.ravel()))
                 
                bak -= temp
                eCon   = np.sum( (bak * bak.conj()).real ) / np.sum( (temp * temp.conj()).real )
                eCon   = np.sqrt(eCon)
                
                eMod   = np.sqrt(eMod / I_norm)
                
                era.update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )
                
                eMods.append(eMod)
                eCons.append(eCon)
        
        if full_output : 
            if rank == 0 :
                info = {}
                info['exits']   = exits
                print 'making Is...'
                info['I']       = np.abs(np.fft.fftn(exits, axes = (-2, -1)))**2
                info['eMod']    = eMods
                info['eCon']    = eCons
                info['heatmap'] = P_heatmap
                if update == 'O': return O, background[0]**2, info
                if update == 'P': return P, background[0]**2, info
                if update == 'OP': return O, P, background[0]**2, info
            else :
                if update == 'OP': 
                    return None, None, None, None
                else :
                    return None, None, None
        else :
            if rank == 0 :
                if update == 'O' : return O, background[0]**2
                if update == 'P' : return P, background[0]**2
                if update == 'OP': return O, P, background[0]**2
            else :
                return None, None



def psup_O(exits, P, R, O_shape, P_heatmap = None, alpha = 1.0e-10):
    OT = np.zeros(O_shape, P.dtype)
    
    # Calculate denominator
    #----------------------
    # but only do this if it hasn't been done already
    # (we must set P_heatmap = None when the probe/coords has changed)
    if P_heatmap is None : 
        P_heatmapT = era.make_P_heatmap(P, R, O_shape)
        P_heatmap  = np.empty_like(P_heatmapT)
        comm.Allreduce([P_heatmapT, MPI.__TypeDict__[P_heatmapT.dtype.char]], \
                       [P_heatmap,  MPI.__TypeDict__[P_heatmap.dtype.char]], \
                       op=MPI.SUM)

    # Calculate numerator
    #--------------------
    for r, exit in zip(R, exits):
        OT[-r[0]:P.shape[0]-r[0], -r[1]:P.shape[1]-r[1]] += exit * P.conj()
         
    # divide
    # here we need to do an all reduce
    #---------------------------------
    O = np.empty_like(OT)
    comm.Allreduce([OT, MPI.__TypeDict__[OT.dtype.char]], \
                   [O, MPI.__TypeDict__[O.dtype.char]],   \
                    op=MPI.SUM)
    O  = O / (P_heatmap + alpha)
    return O, P_heatmap

def psup_P(exits, O, R, O_heatmap = None, alpha = 1.0e-10, centre=False):
    PT = np.zeros(exits[0].shape, exits.dtype)
    
    # Calculate denominator
    #----------------------
    # but only do this if it hasn't been done already
    # (we must set O_heatmap = None when the object/coords has changed)
    if O_heatmap is None : 
        O_heatmapT = np.ascontiguousarray(era.make_O_heatmap(O, R, PT.shape))
        #O_heatmapT = era.make_O_heatmap(O, R, PT.shape) produces a non-contig. array for some reason
        O_heatmap  = np.empty_like(O_heatmapT)
        comm.Allreduce([O_heatmapT, MPI.__TypeDict__[O_heatmapT.dtype.char]], \
                       [O_heatmap,  MPI.__TypeDict__[O_heatmap.dtype.char]], \
                       op=MPI.SUM)

    # Calculate numerator
    #--------------------
    Oc = O.conj()
    for r, exit in zip(R, exits):
        PT += exit * Oc[-r[0]:PT.shape[0]-r[0], -r[1]:PT.shape[1]-r[1]] 
         
    # divide
    #-------
    P = np.empty_like(PT)
    comm.Allreduce([PT, MPI.__TypeDict__[PT.dtype.char]], \
                   [P, MPI.__TypeDict__[P.dtype.char]],   \
                    op=MPI.SUM)
    P  = P / (O_heatmap + alpha)
    
    # roll P to place the centre of mass
    # in the middle of the array
    if centre :
        import scipy.ndimage.measurements
        cm = scipy.ndimage.measurements.center_of_mass((P.conj() * P).real)
        P  = era.multiroll(P, [P.shape[0]//2-np.rint(cm[0]), P.shape[1]//2-np.rint(cm[1])]) 
    return P, O_heatmap

def chunkIt(seq, num):
    splits = np.mgrid[0:len(seq):(num+1)*1J].astype(np.int)
    out    = []
    for i in range(splits.shape[0]-1):
        out.append(seq[splits[i]:splits[i+1]])
    return out
