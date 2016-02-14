import numpy as np
import sys
from itertools import product

import ptychography 
from ptychography.era     import pmod_1, make_exits, update_progress
from ptychography.era_mpi import psup_O, psup_P

def DM_mpi(I, R, P, O, iters, OP_iters = 1, mask = 1, background = None, method = None, hardware = 'cpu', alpha = 1.0e-10, dtype=None, full_output = True):
    """
    MPI variant of ptychography.DM
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
    
    # method 1 or 2 or 3, update O or P or OP
    #---------
    if method == 1 or method == 2 or method == 3 :
        ex_0 = np.empty_like(exits)
        print 'algrithm progress iteration convergence modulus error'
        for i in range(iters) :
            
            # projection 
            # f_i+1 = f_i - Ps f_i + Pm (2 Ps f - f)
            # e0  = Ps f_i
            # e  -= e0           f_i - Ps f_i
            # e0 -= e            Ps f_i - f_i + Ps f_i = 2 Ps f_i - f_i
            # e0  = Pm e0        Pm (2 Ps f_i - f) 
            # e  += e0           f_i - Ps f_i + Pm (2 Ps f_i - f)
            
            # consistency projection 
            if update == 'O': O, P_heatmap = psup_O_1(exits, P, R, O.shape, P_heatmap, alpha = alpha)
            if update == 'P': P, O_heatmap = psup_P_1(exits, O, R, O_heatmap, alpha = alpha)
            if update == 'OP':
                if i % OP_iters[1] == 0 :
                    for j in range(OP_iters[0]):
                        O, P_heatmap = psup_O_1(exits, P, R, O.shape, None, alpha = alpha)
                        P, O_heatmap = psup_P_1(exits, O, R, None, alpha = alpha)
                else :
                        O, P_heatmap = psup_O_1(exits, P, R, O.shape, P_heatmap, alpha = alpha)
            
            ex_0  = make_exits(O, P, R, ex_0)
            
            #exits = exits.copy() - ex_0.copy() + pmod_1(amp, (2*ex_0 - exits).copy(), mask, alpha = alpha)
            exits -= ex_0
            ex_0  -= exits
            ex_0   = pmod_1(amp, ex_0, mask, alpha = alpha)
            exits += ex_0
            
            # metrics
            #--------
            # These are quite expensive, we should only output this every n'th iter to save time
            # f* = Ps f_i = PM (2 Ps f_i - f_i)
            # consistency projection 
            Os = O.copy()
            Ps = P.copy()
            if update == 'O': Os, P_heatmap = psup_O_1(exits, Ps, R, O.shape, P_heatmap, alpha = alpha)
            if update == 'P': Ps, O_heatmap = psup_P_1(exits, Os, R, O_heatmap, alpha = alpha)
            if update == 'OP':
                if i % OP_iters[1] == 0 :
                    for j in range(OP_iters[0]):
                        Os, Ph_t = psup_O_1(exits, Ps, R, O.shape, None, alpha = alpha)
                        Ps, Oh_t = psup_P_1(exits, Os, R, None, alpha = alpha)
                else :
                        Os, P_heatmap = psup_O_1(exits, P, R, O.shape, P_heatmap, alpha = alpha)
            
            ex_0 = make_exits(Os, Ps, R, ex_0)
            eMod = model_error_1(amp, ex_0, mask)
            #eMod = model_error_1(amp, pmod_1(amp, ex_0, mask, alpha=alpha), mask, I_norm)

            if update == 'O' : temp = Os
            if update == 'P' : temp = Ps
            if update == 'OP': temp = np.hstack((Os.ravel(), Ps.ravel()))
            
            bak   -= temp
            eCon   = np.sum( (bak * bak.conj()).real ) / np.sum( (temp * temp.conj()).real )
            eCon   = np.sqrt(eCon)

            eMod = np.sqrt( eMod / I_norm)
            
            update_progress(i / max(1.0, float(iters-1)), 'DM', i, eCon, eMod )

            eMods.append(eMod)
            eCons.append(eCon)
        
            if update == 'O' : bak = Os.copy()
            if update == 'P' : bak = Ps.copy()
            if update == 'OP': bak = np.hstack((Os.ravel().copy(), Ps.ravel().copy()))

        if full_output : 
            info = {}
            info['exits'] = exits
            info['I']     = np.abs(np.fft.fftn(exits, axes = (-2, -1)))**2
            info['eMod']  = eMods
            info['eCon']  = eCons
            info['heatmap']  = P_heatmap
            if update == 'O' : return Os, info
            if update == 'P' : return Ps, info
            if update == 'OP': return Os, Ps, info
        else :
            if update == 'O' : return Os
            if update == 'P' : return Ps
            if update == 'OP': return Os, Ps

    # method 4 or 5 or 6
    #---------
    # update the object with background retrieval
    elif method == 4 or method == 5 or method == 6 :
        if background is None :
            background = np.random.random((I.shape)).astype(dtype)
        else :
            temp       = np.empty(I.shape, dtype = dtype)
            temp[:]    = np.sqrt(background)
            background = temp
        
        ex_0 = np.empty_like(exits)
        b_0  = np.empty_like(background)
        print 'algrithm progress iteration convergence modulus error'
        for i in range(iters) :
            # modulus projection 
            exits, background  = pmod_7(amp, background, exits, mask, alpha = alpha)
            
            background[:] = np.mean(background, axis=0)
            
            # consistency projection 
            if update == 'O': O, P_heatmap = psup_O_1(exits, P, R, O.shape, P_heatmap, alpha = alpha)
            if update == 'P': P, O_heatmap = psup_P_1(exits, O, R, O_heatmap, alpha = alpha)
            if update == 'OP':
                if i % OP_iters[1] == 0 :
                    for j in range(OP_iters[0]):
                        O, P_heatmap = psup_O_1(exits, P, R, O.shape, None, alpha = alpha)
                        P, O_heatmap = psup_P_1(exits, O, R, None, alpha = alpha)
                else :
                        O, P_heatmap = psup_O_1(exits, P, R, O.shape, P_heatmap, alpha = alpha)
            
            b_0[:]  = np.mean(background, axis=0)
            ex_0    = make_exits(O, P, R, ex_0)

            exits      -= ex_0
            background -= b_0
            ex_0       -= exits
            b_0        -= background
            ex_0, b_0   = pmod_7(amp, b_0, ex_0, mask, alpha = alpha)
            exits      += ex_0
            background += b_0

            # metrics
            #--------
            # These are quite expensive, we should only output this every n'th iter to save time
            # f* = Ps f_i = PM (2 Ps f_i - f_i)
            # consistency projection 
            Os = O.copy()
            Ps = P.copy()
            if update == 'O': Os, P_heatmap = psup_O_1(exits, Ps, R, O.shape, P_heatmap, alpha = alpha)
            if update == 'P': Ps, O_heatmap = psup_P_1(exits, Os, R, O_heatmap, alpha = alpha)
            if update == 'OP':
                if i % OP_iters[1] == 0 :
                    for j in range(OP_iters[0]):
                        Os, Ph_t = psup_O_1(exits, Ps, R, O.shape, None, alpha = alpha)
                        Ps, Oh_t = psup_P_1(exits, Os, R, None, alpha = alpha)
                else :
                        Os, P_heatmap = psup_O_1(exits, P, R, O.shape, P_heatmap, alpha = alpha)
            b_0[:]  = np.mean(background, axis=0)
            
            ex_0 = make_exits(Os, Ps, R, ex_0)
            eMod = model_error_1(amp, ex_0, mask, b_0)
            #eMod = model_error_1(amp, pmod_1(amp, ex_0, mask, alpha=alpha), mask, I_norm)

            if update == 'O' : temp = Os
            if update == 'P' : temp = Ps
            if update == 'OP': temp = np.hstack((Os.ravel(), Ps.ravel()))
            
            bak   -= temp
            eCon   = np.sum( (bak * bak.conj()).real ) / np.sum( (temp * temp.conj()).real )
            eCon   = np.sqrt(eCon)
            
            eMod = np.sqrt( eMod / I_norm)

            update_progress(i / max(1.0, float(iters-1)), 'DM', i, eCon, eMod )

            eMods.append(eMod)
            eCons.append(eCon)
        
            if update == 'O' : bak = Os.copy()
            if update == 'P' : bak = Ps.copy()
            if update == 'OP': bak = np.hstack((Os.ravel().copy(), Ps.ravel().copy()))
        
        if full_output : 
            info = {}
            info['exits'] = exits
            info['I']     = np.abs(np.fft.fftn(exits, axes = (-2, -1)))**2
            info['eMod']  = eMods
            info['eCon']  = eCons
            info['heatmap']  = P_heatmap
            if update == 'O' : return O, background**2, info
            if update == 'P' : return P, background**2, info
            if update == 'OP': return O, P, background**2, info
        else :
            if update == 'O':  return O, background**2
            if update == 'P':  return P, background**2
            if update == 'OP': return O, P, background**2



def model_error_1(amp, exits, mask, background = 0):
    exits = np.fft.fftn(exits, axes = (-2, -1))
    M     = np.sqrt((exits.conj() * exits).real + background**2)
    err   = np.sum( mask * (M - amp)**2 ) 
    return err



if __name__ == '__main__' :
    import numpy as np
    import time
    import sys
    #import pyqtgraph as pg
    from era import ERA

    from ptychography.forward_models import forward_sim


    if len(sys.argv) == 2 :
        iters = int(sys.argv[1])
        test = 'all'
    elif len(sys.argv) == 3 :
        iters = int(sys.argv[1])
        test = sys.argv[2]
    else :
        iters = 10
        test = 'all'

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print '\nMaking the forward simulated data...'
    I, R, M, P, O, B = forward_sim(shape_P = (32, 64), shape_O = (128, 128), A = 32, defocus = 1.0e-2,\
                                      photons_pupil = 1, ny = 10, nx = 10, random_offset = None, \
                                      background = None, mask = 100, counting_statistics = False)
    I = np.fft.ifftshift(I, axes=(-2, -1))
    # make the masked pixels bad
    I += 10000. * ~M 
    
    # initial guess for the probe 
    P0 = np.fft.fftshift( np.fft.ifftn( np.abs(np.fft.fftn(P)) ) )
    
    print '\n-------------------------------------------'
    print 'Updating the object on a single cpu core...'

    d0 = time.time()
    Or, info = DM_mpi(I, R, P, None, iters, mask=M, method = 1, alpha=1e-10, dtype='double')
    Or, info = DM_mpi(I, R, P, Or, 50     , mask=M, method = 1, alpha=1e-10, dtype='double')
    d1 = time.time()
    print '\ntime (s):', (d1 - d0) 

    print '\nUpdating the probe on a single cpu core...'
    d0 = time.time()
    Pr, info = DM_mpi(I, R, P0, O, iters, mask=M, method = 2, alpha=1e-10)
    Or, info = DM_mpi(I, R, Pr, O, 50   , mask=M, method = 2, alpha=1e-10, dtype='double')
    d1 = time.time()
    print '\ntime (s):', (d1 - d0) 

    print '\nUpdating the object and probe on a single cpu core...'
    d0 = time.time()
    Or, Pr, info = DM_mpi(I, R, P0, None, iters, mask=M, method = 3, alpha=1e-10)
    d1 = time.time()
    print '\ntime (s):', (d1 - d0) 
    """
    # Single cpu core 
    #----------------
    print '\n-------------------------------------------'
    print 'Updating the object on a single cpu core...'
    try :
        d0 = time.time()
        Or, info = pt.ERA(I, R, P, None, iters, mask=mask, alpha=1e-10, dtype='double')
        d1 = time.time()
        print '\ntime (s):', (d1 - d0) 
    except Exception as e:
        print e

    print '\nUpdating the probe on a single cpu core...'
    try :
        d0 = time.time()
        Pr, info = pt.ERA(I, R, None, O, iters, mask=mask, alpha=1e-10)
        d1 = time.time()
        print '\ntime (s):', (d1 - d0) 
    except Exception as e:
        print e

    print '\nUpdating the object and probe on a single cpu core...'
    try :
        d0 = time.time()
        Or, Pr, info = pt.ERA(I, R, None, None, iters, mask=mask, alpha=1e-10)
        d1 = time.time()
        print '\ntime (s):', (d1 - d0) 
    except Exception as e:
        print e
    """
