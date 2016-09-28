import numpy as np
import sys
from itertools import product

import era

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def DM(I, R, P, O, iters, OP_iters = 1, mask = 1, Fresnel = False, background = None, \
           Pmod_probe = False, probe_centering = False, method = None, hardware = 'cpu', \
           alpha = 1.0e-10, dtype=None, full_output = True, sample_blur = None, verbose = False, \
           output_h5file = None, output_h5group = None, output_h5interval = 1):
    """
    MPI variant of ptychography.DM
    """
    if rank == 0 : print 'DM_mpi v7'    
    method, update, dtype, c_dtype, MPI_dtype, MPI_c_dtype, OP_iters, O, P, amp, background, R, mask, I_norm, N, exits, Fresnel = \
            era.preamble(I, R, P, O, iters, OP_iters, mask, background, method, hardware, alpha, dtype, Fresnel, full_output)

    prop, iprop = era.make_prop(Fresnel, P.shape)
    Pamp = prop(P)
    Pamp = np.sqrt((Pamp.conj() * Pamp).real)

    P_heatmap = None
    O_heatmap = None
    eMods     = []
    eCons     = []
    
    if rank == 0 :
        if update == 'O' : bak = O.copy()
        if update == 'P' : bak = P.copy()
        if update == 'OP': bak = np.hstack((O.ravel().copy(), P.ravel().copy()))
    
    # method 1 or 2 or 3, update O or P or OP
    #---------
    if method == 1 or method == 2 or method == 3 :
        ex_0 = np.empty_like(exits)
        if rank == 0 : print 'algrithm progress iteration convergence modulus error'
        for i in range(iters) :
            
            # projection 
            # f_i+1 = f_i - Ps f_i + Pm (2 Ps f - f)
            # e0  = Ps f_i
            # e  -= e0           f_i - Ps f_i
            # e0 -= e            Ps f_i - f_i + Ps f_i = 2 Ps f_i - f_i
            # e0  = Pm e0        Pm (2 Ps f_i - f) 
            # e  += e0           f_i - Ps f_i + Pm (2 Ps f_i - f)
            
            # consistency projection 
            if update == 'O': O, P_heatmap = era.psup_O(exits, P, R, O.shape, P_heatmap, alpha = alpha, sample_blur=sample_blur)
            if update == 'P': P, O_heatmap = era.psup_P(exits, O, R, O_heatmap, alpha = alpha, sample_blur = sample_blur)
            if update == 'OP':
                if i % OP_iters[1] == 0 :
                    for j in range(OP_iters[0]):
                        O, P_heatmap = era.psup_O(exits, P, R, O.shape, None, alpha, MPI_dtype, MPI_c_dtype, verbose, sample_blur)
                        P, O_heatmap = era.psup_P(exits, O, R, None, alpha, MPI_dtype, MPI_c_dtype)
                    
                    # only centre if both P and O are updated
                    if probe_centering :
                        # get the centre of mass of |P|^2
                        import scipy.ndimage
                        a  = (P.conj() * P).real
                        cm = np.rint(scipy.ndimage.measurements.center_of_mass(a)).astype(np.int) - np.array(a.shape)/2
                        
                        #if rank == 0 : print 'probe cm:', cm
                        
                        # centre P
                        P = era.multiroll(P, -cm)
                        
                        # shift O
                        O = era.multiroll(O, -cm)
                        
                        P_heatmap = O_heatmap = None

                        # because dm remembers the last exits we need to shift them too
                        exits = era.multiroll(exits, [0, -cm[0], -cm[1]])
                        
                else :
                        O, P_heatmap = era.psup_O(exits, P, R, O.shape, P_heatmap, alpha, MPI_dtype, MPI_c_dtype, verbose, sample_blur)
            
            # enforce the modulus of the far-field probe 
            if Pmod_probe is not None and i < Pmod_probe :
                P = era.Pmod_P(Pamp, P, mask, alpha, prop = (prop, iprop))

            ex_0  = era.make_exits(O, P, R, ex_0)
            
            #exits = exits.copy() - ex_0.copy() + pmod_1(amp, (2*ex_0 - exits).copy(), mask, alpha = alpha)
            exits -= ex_0
            ex_0  -= exits
            #ex_0   = era.pmod_1(amp, ex_0, mask, alpha = alpha)
            ex_0   = era.pmod_1(amp, ex_0, mask, alpha = alpha, eMod_calc = False, prop = (prop, iprop))
            exits += ex_0

            
            # metrics
            #--------
            # These are quite expensive, we should only output this every n'th iter to save time
            # f* = Ps f_i = PM (2 Ps f_i - f_i)
            # consistency projection 
            Os = O.copy()
            Ps = P.copy()
            if update == 'O': Os, P_heatmap = era.psup_O(exits, Ps, R, O.shape, P_heatmap, alpha = alpha, sample_blur = sample_blur)
            if update == 'P': Ps, O_heatmap = era.psup_P(exits, Os, R, O_heatmap, alpha = alpha)
            if update == 'OP':
                if i % OP_iters[1] == 0 :
                    for j in range(OP_iters[0]):
                        Os, Ph_t = era.psup_O(exits, Ps, R, O.shape, None, alpha = alpha, sample_blur = sample_blur)
                        Ps, Oh_t = era.psup_P(exits, Os, R, None, alpha = alpha)
                    
                    # only centre if both P and O are updated
                    if probe_centering :
                        # get the centre of mass of |P|^2
                        import scipy.ndimage
                        a  = (Ps.conj() * Ps).real
                        cm = np.rint(scipy.ndimage.measurements.center_of_mass(a)).astype(np.int) - np.array(a.shape)/2
                        
                        #if rank == 0 : print 'probe cm:', cm
                        
                        # centre P
                        Ps = era.multiroll(Ps, -cm)
                        
                        # shift O
                        Os = era.multiroll(Os, -cm)
                        
                else :
                        Os, P_heatmap = era.psup_O(exits, P, R, O.shape, P_heatmap, alpha = alpha, sample_blur = sample_blur)
            
            # enforce the modulus of the far-field probe 
            if Pmod_probe is not None and i < Pmod_probe :
                Ps = era.Pmod_P(Pamp, Ps, mask, alpha, prop = (prop, iprop))

            ex_0 = era.make_exits(Os, Ps, R, ex_0)
            eMod = model_error_1(amp, ex_0, mask, prop=(prop, iprop))
            #eMod = model_error_1(amp, pmod_1(amp, ex_0, mask, alpha=alpha), mask, I_norm)
            
            eMod   = comm.reduce(eMod, op=MPI.SUM)
            
            if rank == 0 :
                if update == 'O' : temp = Os
                if update == 'P' : temp = Ps
                if update == 'OP': temp = np.hstack((Os.ravel(), Ps.ravel()))
                
                bak   -= temp
                eCon   = np.sum( (bak * bak.conj()).real ) / np.sum( (temp * temp.conj()).real )
                eCon   = np.sqrt(eCon)

                eMod = np.sqrt( eMod / I_norm)
                
                era.update_progress(i / max(1.0, float(iters-1)), 'DM', i, eCon, eMod )

                eMods.append(eMod)
                eCons.append(eCon)
            
                if update == 'O' : bak = Os.copy()
                if update == 'P' : bak = Ps.copy()
                if update == 'OP': bak = np.hstack((Os.ravel().copy(), Ps.ravel().copy()))
                
                # output O, P and eMod 
                ######################
                if (output_h5file is not None) and (i % output_h5interval == 0) :
                    import h5py
                    f = h5py.File(output_h5file)
                    key = output_h5group + '/O'
                    if key in f :
                        del f[key]
                    f[key] = Os
                    
                    key = output_h5group + '/P'
                    if key in f :
                        del f[key]
                    f[key] = Ps
                    
                    key = output_h5group + '/eMod'
                    if key in f :
                        del f[key]
                    f[key] = np.array(eMods)
                    f.close()

    # method 4 or 5 or 6
    #---------
    # update the object with background retrieval
    elif method == 4 or method == 5 or method == 6 :
        ex_0 = np.empty_like(exits)
        b_0  = np.empty_like(background)
        if rank == 0 : print 'algrithm progress iteration convergence modulus error'
        for i in range(iters) :
            # modulus projection 
            exits, background  = era.pmod_7(amp, background, exits, mask, alpha = alpha)
            
            # consistency projection 
            if update == 'O': O, P_heatmap = era.psup_O(exits, P, R, O.shape, P_heatmap, alpha = alpha, sample_blur = sample_blur)
            if update == 'P': P, O_heatmap = era.psup_P(exits, O, R, O_heatmap, alpha = alpha, sample_blur = sample_blur)
            if update == 'OP':
                if i % OP_iters[1] == 0 :
                    for j in range(OP_iters[0]):
                        O, P_heatmap = era.psup_O(exits, P, R, O.shape, None, alpha = alpha)
                        P, O_heatmap = era.psup_P(exits, O, R, None, alpha = alpha)
                    
                    # only centre if both P and O are updated
                    if probe_centering :
                        # get the centre of mass of |P|^2
                        import scipy.ndimage
                        a  = (P.conj() * P).real
                        cm = np.rint(scipy.ndimage.measurements.center_of_mass(a)).astype(np.int) - np.array(a.shape)/2
                        
                        if rank == 0 : print 'probe cm:', cm
                        
                        # centre P
                        P = era.multiroll(P, -cm)
                        
                        # shift O
                        O = era.multiroll(O, -cm)
                        
                        P_heatmap = O_heatmap = None
                        
                        # because dm remembers the last exits we need to shift them too
                        exits = era.multiroll(exits, [0, -cm[0], -cm[1]])
                else :
                        O, P_heatmap = era.psup_O(exits, P, R, O.shape, P_heatmap, alpha = alpha, sample_blur = sample_blur)
            
            # enforce the modulus of the far-field probe 
            if Pmod_probe is not None and i < Pmod_probe :
                P = era.Pmod_P(Pamp, P, mask, alpha, prop = (prop, iprop))

            backgroundT  = np.mean(background, axis=0)
            backgroundTT = np.empty_like(backgroundT)
            comm.Allreduce([backgroundT, MPI_dtype], \
                           [backgroundTT,  MPI_dtype], \
                           op=MPI.SUM)
            b_0[:] = backgroundTT / float(size)

            #b_0[:]  = np.mean(background, axis=0)
            ex_0    = era.make_exits(O, P, R, ex_0)

            exits      -= ex_0
            background -= b_0
            ex_0       -= exits
            b_0        -= background
            ex_0, b_0   = era.pmod_7(amp, b_0, ex_0, mask, alpha = alpha)
            exits      += ex_0
            background += b_0

            # metrics
            #--------
            # These are quite expensive, we should only output this every n'th iter to save time
            # f* = Ps f_i = PM (2 Ps f_i - f_i)
            # consistency projection 
            Os = O.copy()
            Ps = P.copy()
            if update == 'O': Os, P_heatmap = era.psup_O(exits, Ps, R, O.shape, P_heatmap, alpha = alpha, sample_blur = sample_blur)
            if update == 'P': Ps, O_heatmap = era.psup_P(exits, Os, R, O_heatmap, alpha = alpha)
            if update == 'OP':
                if i % OP_iters[1] == 0 :
                    for j in range(OP_iters[0]):
                        Os, Ph_t = era.psup_O(exits, Ps, R, O.shape, None, alpha = alpha, sample_blur = sample_blur)
                        Ps, Oh_t = era.psup_P(exits, Os, R, None, alpha = alpha)
                    
                    # only centre if both P and O are updated
                    if probe_centering :
                        # get the centre of mass of |P|^2
                        import scipy.ndimage
                        a  = (Ps.conj() * Ps).real
                        cm = np.rint(scipy.ndimage.measurements.center_of_mass(a)).astype(np.int) - np.array(a.shape)/2
                        
                        if rank == 0 : print 'probe cm:', cm
                        
                        # centre P
                        Ps = era.multiroll(Ps, -cm)
                        
                        # shift O
                        Os = era.multiroll(Os, -cm)
                        
                else :
                        Os, P_heatmap = era.psup_O(exits, P, R, O.shape, P_heatmap, alpha = alpha, sample_blur = sample_blur)
            
            # enforce the modulus of the far-field probe 
            if Pmod_probe is not None and i < Pmod_probe :
                Ps = era.Pmod_P(Pamp, Ps, mask, alpha, prop = (prop, iprop))

            backgroundT  = np.mean(background, axis=0)
            backgroundTT = np.empty_like(backgroundT)
            comm.Allreduce([backgroundT, MPI_dtype], \
                           [backgroundTT,  MPI_dtype], \
                           op=MPI.SUM)
            b_0[:] = backgroundTT / float(size)
            
            ex_0 = era.make_exits(Os, Ps, R, ex_0)
            eMod = model_error_1(amp, ex_0, mask, b_0)
            #eMod = model_error_1(amp, pmod_1(amp, ex_0, mask, alpha=alpha), mask, I_norm)
             
            eMod   = comm.reduce(eMod, op=MPI.SUM)
               
            if rank == 0 :
                if update == 'O' : temp = Os
                if update == 'P' : temp = Ps
                if update == 'OP': temp = np.hstack((Os.ravel(), Ps.ravel()))
                
                bak   -= temp
                eCon   = np.sum( (bak * bak.conj()).real ) / np.sum( (temp * temp.conj()).real )
                eCon   = np.sqrt(eCon)
                
                eMod = np.sqrt( eMod / I_norm)

                era.update_progress(i / max(1.0, float(iters-1)), 'DM', i, eCon, eMod )

                eMods.append(eMod)
                eCons.append(eCon)
            
                if update == 'O' : bak = Os.copy()
                if update == 'P' : bak = Ps.copy()
                if update == 'OP': bak = np.hstack((Os.ravel().copy(), Ps.ravel().copy()))
        
    """
    # This should not be necessary but it crashes otherwise
    exits = era.make_exits(Os, Ps, R, exits)
    I = np.fft.fftshift(np.abs(np.fft.fftn(exits, axes = (-2, -1)))**2, axes = (-2, -1))
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
        info['eCon']    = eCons
        info['heatmap'] = P_heatmap
        if background is not None :
            info['background'] = np.fft.fftshift(b_0[0])**2
        return O, P, info
    else :
        return None, None, None
    """

    # This should not be necessary but it crashes otherwise
    #I = np.fft.fftshift(np.abs(np.fft.fftn(exits, axes = (-2, -1)))**2, axes = (-2, -1))
    #exits = era.make_exits(Os, Ps, R, exits)
    per_diff_eMod = era.model_error_per_diff(amp, ex_0, mask, background = 0, prop = prop)
       
    I = np.fft.fftshift(np.abs(prop(ex_0))**2, axes = (-2,-1))
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
        return Os, Ps, info
    else :
        return None, None, None



def model_error_1(amp, exits, mask, background = 0, prop = None):
    if prop is None :
        exits = np.fft.fftn(exits, axes = (-2, -1))
    else :
        exits = prop[0](exits)
   # exits = np.fft.fftn(exits, axes = (-2, -1))
    M     = np.sqrt((exits.conj() * exits).real + background**2)
    err   = np.sum( mask * (M - amp)**2 ) 
    return err



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
                                              photons_pupil = 1, ny = 10, nx = 10, random_offset = 5, \
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
            print 'Updating the object on ',size ,' cpu cores...'
            d0 = time.time()

        Or, info = DM(I, R, P, None, iters, mask=M, method = 1, hardware = 'mpi', alpha=1e-10, dtype='double')
        
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

        Pr, info = DM_mpi(I, R, P0, O, iters, mask=M, method = 2, hardware = 'mpi', alpha=1e-10)
        
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

        Or, Pr, info = DM_mpi(I, R, P0, None, iters, mask=M, method = 3, hardware = 'mpi', alpha=1e-10)

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

        Or, info = DM_mpi(I, R, P, None, iters, mask=M, method = 4, hardware = 'mpi', alpha=1e-10, dtype='double')

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

        Pr, info = DM_mpi(I, R, P0, O, iters, mask=M, method = 5, hardware = 'mpi', alpha=1e-10)

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

        Or, Pr, info = DM_mpi(I, R, P0, None, iters, mask=M, method = 6, hardware = 'mpi', alpha=1e-10)

        if rank == 0 : 
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
            
            write_cxi(I, info['I'], P, Pr, O, Or, \
                      R, None, B, info['background'], M, info['eMod'], fnam = 'output_method6.cxi')
