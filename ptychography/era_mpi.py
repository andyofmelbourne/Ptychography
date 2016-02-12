import numpy as np
import sys

import era

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def ERA_mpi(I, R, P, O, iters, OP_iters = 1, mask = 1, background = None, method = None, hardware = 'cpu', alpha = 1.0e-10, dtype=None, full_output = True):
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

    # make our exit waves
    exits     = era.make_exits(O, P, R)

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
            exits, eMod = era.pmod_1(amp, exits, mask, alpha = alpha, eMod_calc = True)
            
            # consistency projection 
            if update == 'O': O, P_heatmap = psup_O(exits, P, R, O.shape, P_heatmap, alpha, MPI_dtype, MPI_c_dtype)
            if update == 'P': P, O_heatmap = psup_P(exits, O, R, O_heatmap, alpha, MPI_dtype, MPI_c_dtype)
            if update == 'OP':
                if i % OP_iters[1] == 0 :
                    for j in range(OP_iters[0]):
                        O, P_heatmap = psup_O(exits, P, R, O.shape, None, alpha, MPI_dtype, MPI_c_dtype)
                        P, O_heatmap = psup_P(exits, O, R, None, alpha, MPI_dtype, MPI_c_dtype)
                        # rescale
                        #Omax = np.max(np.abs(O))
                        #O = O / Omax
                        #P = P * Omax

                else :
                    O, P_heatmap = psup_O(exits, P, R, O.shape, P_heatmap, alpha, MPI_dtype, MPI_c_dtype)
            
            exits = era.make_exits(O, P, R, exits)
            
            # metrics
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

    # method 4 or 5
    #---------
    # update the object with background retrieval
    elif method == 4 or method == 5 or method == 6 :
        if background is None :
            background = np.random.random((exits.shape)).astype(dtype)
        else :
            temp       = np.empty(exits.shape, dtype = dtype)
            temp[:]    = np.sqrt(background)
            background = temp
        
        if rank == 0 : print 'algrithm progress iteration convergence modulus error'
        for i in range(iters) :
            if rank == 0 : 
                if update == 'O' : bak = O.copy()
                if update == 'P' : bak = P.copy()
                if update == 'OP': bak = np.hstack((O.ravel().copy(), P.ravel().copy()))
            
            # modulus projection 
            exits, background, eMod = era.pmod_7(amp, background, exits, mask, alpha = alpha, eMod_calc = True)
            
            # consistency projection 
            if update == 'O': O, P_heatmap = psup_O(exits, P, R, O.shape, P_heatmap, alpha, MPI_dtype, MPI_c_dtype)
            if update == 'P': P, O_heatmap = psup_P(exits, O, R, O_heatmap, alpha, MPI_dtype, MPI_c_dtype)
            if update == 'OP':
                if i % OP_iters[1] == 0 :
                    for j in range(OP_iters[0]):
                        O, P_heatmap = psup_O(exits, P, R, O.shape, None, alpha, MPI_dtype, MPI_c_dtype)
                        P, O_heatmap = psup_P(exits, O, R, None, alpha, MPI_dtype, MPI_c_dtype)
                else :
                    O, P_heatmap = psup_O(exits, P, R, O.shape, P_heatmap, alpha, MPI_dtype, MPI_c_dtype)

            #background[:] = np.mean(background, axis=0)
            backgroundT = np.mean(background, axis=0)
            backgroundTT = np.empty_like(backgroundT)
            #comm.Allreduce([backgroundT, MPI.__TypeDict__[backgroundT.dtype.char]], \
            #               [backgroundTT,  MPI.__TypeDict__[backgroundTT.dtype.char]], \
            #               op=MPI.SUM)
            comm.Allreduce([backgroundT, MPI_dtype], \
                           [backgroundTT,  MPI_dtype], \
                           op=MPI.SUM)
            background[:] = backgroundTT / float(size)
            
            exits = era.make_exits(O, P, R, exits)
            
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
                
                era.update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )
                
                eMods.append(eMod)
                eCons.append(eCon)
        
        if full_output : 
            if rank == 0 :
                info = {}
                info['exits']   = exits
                info['I']       = np.abs(np.fft.fftn(exits, axes = (-2, -1)))**2
                info['eMod']    = eMods
                info['eCon']    = eCons
                info['heatmap'] = P_heatmap
                if update == 'O': return O, background**2, info
                if update == 'P': return P, background**2, info
                if update == 'OP': return O, P, background**2, info
            else :
                if update == 'OP': 
                    return None, None, None, None
                else :
                    return None, None, None
        else :
            if rank == 0 :
                if update == 'O' : return O, background**2
                if update == 'P' : return P, background**2
                if update == 'OP': return O, P, background**2
            else :
                return None, None



def psup_O(exits, P, R, O_shape, P_heatmap = None, alpha = 1.0e-10, MPI_dtype = MPI.DOUBLE, MPI_c_dtype = MPI.DOUBLE_COMPLEX):
    OT = np.zeros(O_shape, P.dtype)
    
    # Calculate denominator
    #----------------------
    # but only do this if it hasn't been done already
    # (we must set P_heatmap = None when the probe/coords has changed)
    if P_heatmap is None : 
        P_heatmapT = era.make_P_heatmap(P, R, O_shape)
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
    O  = O / (P_heatmap + alpha)
    return O, P_heatmap

def psup_P(exits, O, R, O_heatmap = None, alpha = 1.0e-10, MPI_dtype = MPI.DOUBLE, MPI_c_dtype = MPI.DOUBLE_COMPLEX):
    PT = np.zeros(exits[0].shape, exits.dtype)
    
    # Calculate denominator
    #----------------------
    # but only do this if it hasn't been done already
    # (we must set O_heatmap = None when the object/coords has changed)
    if O_heatmap is None : 
        O_heatmapT = np.ascontiguousarray(era.make_O_heatmap(O, R, PT.shape))
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

def chunkIt(seq, num):
    splits = np.mgrid[0:len(seq):(num+1)*1J].astype(np.int)
    out    = []
    for i in range(splits.shape[0]-1):
        out.append(seq[splits[i]:splits[i+1]])
    return out
