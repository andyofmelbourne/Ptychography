import numpy as np
import sys

import era

def ERA_gpu(I, R, P, O, iters, OP_iters = 1, mask = 1, background = None, method = None, hardware = 'cpu', alpha = 1.0e-10, dtype=None, full_output = True):
    """
    GPU variant of ptychography.ERA
    """
    if method == None :
        if O is None and P is None :
            method = 3
        elif O is None :
            method = 1
        elif P is None :
            method = 2
    
    if method == 1 or method == 4: 
        update = 'O'
    elif method == 2 or method == 5: 
        update = 'P'
    elif method == 3 or method == 6: 
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
    exits     = era.make_exits(O, P, R)
    P_heatmap = None
    O_heatmap = None
    eMods     = []
    eCons     = []

    # subtract an overall offset from R's
    R[:, 0] -= R[:, 0].max()
    R[:, 1] -= R[:, 1].max()
    
    # set up the gpu
    #---------------
    #---------------
    import pyfft
    import pyopencl
    import pyopencl.array
    from   pyfft.cl import Plan
    import pyopencl.clmath 

    # get the CUDA platform
    print 'opencl platforms found:'
    platforms = pyopencl.get_platforms()
    for p in platforms:
        print '\t', p.name
        if p.name == 'NVIDIA CUDA':
            platform = p
            print '\tChoosing', p.name

    # get one of the gpu's device id
    print '\nopencl devices found:'
    devices = platform.get_devices()
    for d in devices:
        print '\t', d.name

    print '\tChoosing', devices[0].name
    device = devices[0]
    
    # create a context for the device
    context = pyopencl.Context([device])
    
    # create a command queue for the device
    queue = pyopencl.CommandQueue(context)
    
    # make a plan for the ffts
    plan = Plan(I[0].shape, dtype=c_dtype, queue=queue)

    exits_g = pyopencl.array.to_device(queue, np.ascontiguousarray(exits))
    amp_g   = pyopencl.array.to_device(queue, np.ascontiguousarray(amp))
    if mask is not 1 :
        mask_g    = np.empty(I.shape, dtype=np.int8)
        mask_g[:] = mask.astype(np.int8)*2 - 1
        mask_g    = pyopencl.array.to_device(queue, np.ascontiguousarray(mask_g))
    else :
        mask_g  = 1

    # method 1, 2 or 3, update O or P or 'OP'
    #---------
    if method == 1 or method == 2 or method == 3:
        
        print 'algrithm progress iteration convergence modulus error'
        for i in range(iters) :
            if update == 'O' : bak = O.copy()
            if update == 'P' : bak = P.copy()
            if update == 'OP': bak = np.hstack((O.ravel().copy(), P.ravel().copy()))
		
            E_bak        = exits.copy()
            
            # modulus projection 
            exits_g.set(exits)
            exits        = pmod_4(amp_g, exits_g, plan, mask_g, alpha = alpha).get()
            
            E_bak       -= exits

            # consistency projection 
            if update == 'O' : O, P_heatmap = era.psup_O_1(exits, P, R, O.shape, P_heatmap, alpha = alpha)
            if update == 'P' : P, O_heatmap = era.psup_P_1(exits, O, R, O_heatmap, alpha = alpha)
            if update == 'OP': 
                for j in range(OP_iters):
                    O, P_heatmap = era.psup_O_1(exits, P, R, O.shape, None, alpha = alpha)
                    P, O_heatmap = era.psup_P_1(exits, O, R, None, alpha = alpha)
            
            exits = era.make_exits(O, P, R, exits)
            
            # metrics
            if update == 'O' : temp = O
            if update == 'P' : temp = P
            if update == 'OP': temp = np.hstack((O.ravel(), P.ravel()))
            
            bak -= temp
            eCon   = np.sum( (bak * bak.conj()).real ) / np.sum( (temp * temp.conj()).real )
            eCon   = np.sqrt(eCon)
            
            eMod   = np.sum( (E_bak * E_bak.conj()).real ) / I_norm
            eMod   = np.sqrt(eMod)
            
            era.update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )

            eMods.append(eMod)
            eCons.append(eCon)
        
        if full_output : 
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
            if update == 'O' : return O
            if update == 'P' : return P
            if update == 'OP': return O, P

    # method 4, 5 or 6, update O or P or 'OP' with background
    #---------
    if method == 4 or method == 5 or method == 6:
        if background is None :
            background = np.random.random((I.shape)).astype(dtype)
        else :
            temp       = np.empty(I.shape, dtype = dtype)
            temp[:]    = np.sqrt(background)
            background = temp

        background_g   = pyopencl.array.to_device(queue, np.ascontiguousarray(background))
        
        print 'algrithm progress iteration convergence modulus error'
        print 'iters:', iters
        for i in range(iters) :
            if update == 'O' : bak = O.copy()
            if update == 'P' : bak = P.copy()
            if update == 'OP': bak = np.hstack((O.ravel().copy(), P.ravel().copy()))
		
            E_bak        = exits.copy()
            
            # modulus projection 
            exits_g.set(exits)
            background_g.set(background)
            #exits        = pmod_4(amp_g, exits_g, plan, mask_g, alpha = alpha).get()
            exits_g, background_g = pmod_back(amp_g, background_g, exits_g, plan, mask_g, alpha = alpha)
            exits      = exits_g.get()
            background = background_g.get()
            
            E_bak       -= exits

            # consistency projection 
            if update == 'O' : O, P_heatmap = era.psup_O_1(exits, P, R, O.shape, P_heatmap, alpha = alpha)
            if update == 'P' : P, O_heatmap = era.psup_P_1(exits, O, R, O_heatmap, alpha = alpha)
            if update == 'OP': 
                for j in range(OP_iters):
                    O, P_heatmap = era.psup_O_1(exits, P, R, O.shape, None, alpha = alpha)
                    P, O_heatmap = era.psup_P_1(exits, O, R, None, alpha = alpha)
            
            exits = era.make_exits(O, P, R, exits)
            
            background[:] = np.mean(background, axis=0)
            
            # metrics
            if update == 'O' : temp = O
            if update == 'P' : temp = P
            if update == 'OP': temp = np.hstack((O.ravel(), P.ravel()))
            
            bak -= temp
            eCon   = np.sum( (bak * bak.conj()).real ) / np.sum( (temp * temp.conj()).real )
            eCon   = np.sqrt(eCon)
            
            eMod   = np.sum( (E_bak * E_bak.conj()).real ) / I_norm
            eMod   = np.sqrt(eMod)
            
            era.update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )

            eMods.append(eMod)
            eCons.append(eCon)
        
        if full_output : 
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
            if update == 'O' : return O
            if update == 'P' : return P
            if update == 'OP': return O, P


def pmod_4(amp, exits, plan, mask = 1, alpha = 1.0e-10):
    plan.execute(exits.data, batch = exits.shape[0])
    exits = Pmod_4(amp, exits, mask = mask, alpha = alpha)
    plan.execute(exits.data, batch = exits.shape[0], inverse = True)
    return exits
    
def Pmod_4(amp, exits, mask = 1, alpha = 1.0e-10):
    import pyopencl.array
    if mask is 1 :
        exits  = exits * amp / (abs(exits) + alpha)
    else :
        #exits  = mask * exits * amp / (abs(exits) + alpha)
        exits2 = exits * amp / (abs(exits) + alpha)
        pyopencl.array.if_positive(mask, exits2, exits, out = exits)
        #exits.mul_add(mask * amp / (abs(exits) + alpha), (1 - mask), exits)
    return exits

def pmod_back(amp, background, exits, plan, mask = 1, alpha = 1.0e-10):
    plan.execute(exits.data, batch = exits.shape[0])
    exits, background = Pmod_back(amp, background, exits, mask = mask, alpha = alpha)
    plan.execute(exits.data, batch = exits.shape[0], inverse = True)
    return exits, background
    
def Pmod_back(amp, background, exits, mask = 1, alpha = 1.0e-10):
    import pyopencl.array
    import pyopencl.clmath
    M = amp / pyopencl.clmath.sqrt((exits.conj() * exits).real + background**2 + alpha)
    if mask is 1 :
        exits      *= M
        background *= M
    else :
        M *= mask
        exits2     = exits * M
        background = background * M
        pyopencl.array.if_positive(mask, exits2, exits, out = exits)
    return exits, background
