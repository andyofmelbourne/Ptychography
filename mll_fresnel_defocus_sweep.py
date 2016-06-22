import numpy as np
import h5py
import scipy.constants as sc
import time
from scipy import ndimage

import Ptychography.ptychography.era as era
from Ptychography import DM
from Ptychography import ERA
from Ptychography import utils
from Ptychography import write_cxi

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def Psup(I, R, P, O, iters, OP_iters, mask, background, method, hardware, alpha, dtype, full_output, verbose, sample_blur):
    # set the probe to the whitefield
    #if rank == 0 : 
    #    P = whitefield + 0J
    #else :
    #    P = None
    
    method, update, dtype, c_dtype, MPI_dtype, MPI_c_dtype, OP_iters, O, P, amp, Pamp, background, R, mask, I_norm, N, exits = \
            era.preamble(I, R, P, O, iters, OP_iters, mask, background, method, hardware, alpha, dtype, full_output)
    
    O.fill(1.)
    
    # set the exit waves to the sqrt(I)
    exits = np.fft.fftshift(amp) + 0J  
    
    P_heatmap    = None
    O_heatmap    = None
    O, P_heatmap = era.psup_O(exits, P, R, O.shape, P_heatmap, alpha, MPI_dtype, MPI_c_dtype, verbose, sample_blur)
    return O, P


def downsample(I, n = 2):
    out = np.sum(I.reshape((I.shape[0], I.shape[1]/n, n)), axis=-1)
    out = np.sum(out.T.reshape((I.shape[1]/n, I.shape[0]/n, n)), axis=-1).T
    return out

def zero_pad_to_nearest_pow2(diff, shape_new = None, verbose = False, fillvalue=0.):
    """
    find the smallest power of 2 that 
    fits each dimension of the diffraction
    pattern then zero pad keeping the zero
    pixel centred
    """
    if shape_new is None :
        shape_new = []
        for s in diff.shape:
            n = 0
            while 2**n < s :
                n += 1
            shape_new.append(2**n)

    if verbose : print '\nreshaping:', diff.shape, '-->', shape_new
    diff_new = np.ones(tuple(shape_new), dtype=diff.dtype) * fillvalue
    diff_new[:diff.shape[0], :diff.shape[1]] = diff

    # roll the axis to keep N / 2 at N'/2
    for i in range(len(shape_new)):
        diff_new = np.roll(diff_new, shape_new[i]/2 - diff.shape[i] / 2, i)
    return diff_new

def crop_pad_downsample(array, crop_ijs, crop_shape, pad, n, fillvalue = 0):
    if (crop_ijs is not None) and (crop_shape is not None) :
        out = array[crop_ijs].reshape(crop_shape)
    
    if (pad is not None) :
        out = zero_pad_to_nearest_pow2(out, pad, fillvalue = fillvalue)
    
    if (n is not None) :
        out = downsample(out, n)
    
    return out

def get_mask(f, params):
    """
    We need to get the bad pixel mask.
    Then crop to the pupil function if requested.
    Then pad to 2^n if requested.
    Then down sample if requested.
    Also return the function that bins / crops / pads.
    """
    # mask hot / dead pixels
    print '\nMasking hot and dead pixels', 
    bit_mask = f['entry_1/instrument_1/detector_1/mask'].value
    
    # hot (4) and dead (8) pixels
    mask     = ~np.bitwise_and(bit_mask, 4 + 8).astype(np.bool) 
    mask_old = mask.copy()
    
    print np.sum(~mask), 'pixels out of ', mask.size, 'found'
    
    if params['input']['crop_to_pupil'] is True :
        # the pupil
        pupil = np.bitwise_and(bit_mask, 2**10).astype(np.bool) 
        
        cropped    = utils.crop_to_nonzero(pupil)
        crop_shape = cropped.shape
        
        print '\ncroping the frames to the smallest rectangle that contains the pupil function'
        print np.sum(pupil), ' pixels in the pupil mask'
        print pupil.shape,'-->', cropped.shape
        
        crop_ijs  = np.where(pupil)
        mask_norm = mask[crop_ijs].reshape(crop_shape)
    else :
        crop_shape = crop_ijs = None
        mask_norm  = mask.copy()

    if params['input']['pad_to_nearest_pow2'] :
        mask_norm = zero_pad_to_nearest_pow2(mask_norm, shape_new = None, verbose = True, fillvalue=0)
        pad       = mask_norm.shape
    else :
        pad       = None
    
    if params['input']['downsample'] is not None :
        mask_norm = downsample(mask_norm.astype(np.int), params['input']['downsample'])
        
        mask_norm[mask_norm == 0] = 1
        mask_norm = mask_norm.astype(np.float)
        
        n = params['input']['downsample']
    else :
        mask_norm = 1.    
        n = None
    
    I_crop_pad_downsample = lambda x : crop_pad_downsample(x * mask_old, crop_ijs, crop_shape, pad, n, fillvalue = 0.) / mask_norm 
    
    # enforce 0's in the padded region with fillvalue = 1
    mask = crop_pad_downsample(mask, crop_ijs, crop_shape, pad, n, fillvalue = 1)
    mask = mask > 0
    
    return mask, I_crop_pad_downsample 


def get_Rs(f, mask, params):
    # get the sample positions
    fast_axis   = f['entry_1/instrument_1/motor_positions/scan_axes/Fast axis/name'].value
    slow_axis   = f['entry_1/instrument_1/motor_positions/scan_axes/Slow axis/name'].value
    mll1_name   = f['entry_1/sample_1/name'].value
    mll2_name   = f['entry_1/sample_2/name'].value
    sample_name = f['entry_1/sample_3/name'].value
    z           = f['entry_1/instrument_1/detector_1/distance'].value
    
    # get the beam energy and wavelength
    E    = f['entry_1/instrument_1/source_1/energy'].value 
    lamb = sc.h * sc.c / E
    
    # get the pixel size and q step
    du = [f['entry_1/instrument_1/detector_1/x_pixel_size'].value, \
          f['entry_1/instrument_1/detector_1/x_pixel_size'].value]
    du = np.array(du) * params['input']['downsample']
    dq = du / (lamb * z)
    
    # get the grid steps 
    steps = [f['entry_1/instrument_1/motor_positions/scan_axes/Slow axis/Steps'].value, 
             f['entry_1/instrument_1/motor_positions/scan_axes/Fast axis/Steps'].value]
                    
    print '\n'
    print 'MLL1 name                 :', mll1_name
    print 'MLL2 name                 :', mll2_name
    print 'Sample name               :', sample_name
    print 'Beam energy (keV)         : {0:.2f}'.format( 1.0e-3 * E / sc.e )
    print 'Beam energy (J)           : {0:.2e}'.format( E )
    print 'Wavelength (m)            : {0:.2e}'.format( lamb )
    print 'Steps slow x fast (steps) :', steps[0], 'x', steps[1]
    print 'pixel dimensions (um)     :', 1.0e6 * du, 'm'
    print 'dq (A-1)                  :', 1.0e10*dq, 'm-1'
    
    # Interferometer
    if params['input']['interferometer'] == True :
        slow_axis = 'IFMY'
        fast_axis = 'IFMX'
        print '\nLoading encoded fast and slow axis values', fast_axis, 'and', slow_axis
        X = f['entry_1/instrument_1/motor_positions/scan_axes/Interferometer/IFMX'].value
        Y = f['entry_1/instrument_1/motor_positions/scan_axes/Interferometer/IFMY'].value
        X *= 1.0e-9
        Y *= -1.0e-9
    # get the XPZT and YPZT values
    else :
        print '\nLoading encoded fast and slow axis values', fast_axis, 'and', slow_axis
        X = f['entry_1/instrument_1/motor_positions/scan_axes/Fast axis/name'].value
        Y = f['entry_1/instrument_1/motor_positions/scan_axes/Slow axis/name'].value
        X = f['entry_1/instrument_1/motor_positions/scan_axes/Fast axis/POSITION'].value
        Y = f['entry_1/instrument_1/motor_positions/scan_axes/Slow axis/POSITION'].value
    
    Rindex  = np.arange(len(X))
    R       = np.zeros((len(X), 3), dtype=np.float)
    R[:, 0] = Y
    R[:, 1] = X
    R[:, 2] = z
    
    # discard bad readings
    good_vals = ~(np.isnan(X) * np.isnan(Y))
    bad_vals  =  np.sum(~good_vals)
    if bad_vals > 0:
        print '\nBad readings:', bad_vals, 'out of:', len(X)
        print 'Discarding bad readings'
    good_vals = np.where(good_vals)[0]
    
    # discard unwanted frames
    #########################
    # grid of points
    if params['input']['startij_stopij'] is -1 or params['input']['startij_stopij'] is None :
        steps_subset = range(len(R))
    else :
        steps_subset = []
        for i in range(params['input']['startij_stopij'][0], params['input']['startij_stopij'][2], 1):
            for j in range(params['input']['startij_stopij'][1], params['input']['startij_stopij'][3], 1):
                step = steps[1] * i + j
                if step in good_vals :
                    steps_subset.append( step )
        print '\nLoading a grid of points in bounded by:', params['input']['startij_stopij'][:2], \
                'and',params['input']['startij_stopij'][2:]
        print 'of these,', len(steps_subset),'are valid frames.'
    steps_subset = np.array(steps_subset)

    # skip every n-frames
    if params['input']['every_n_frames'] is -1 or params['input']['every_n_frames'] is None :
        every = 1
    else :
        every = params['input']['every_n_frames'] 
    print '\nLoading every', every, 'frames'
    
    steps_subset = steps_subset[::every]
        
    # update index
    Rindex = Rindex[steps_subset] 
    R      = R[steps_subset] 

    # print average step sizes
    print '\nThis is a slow scan of:', slow_axis
    print 'From:', R[0,0], 'to:', R[-1,0], 'steps:', steps[0], 'range:', np.abs(R[-1,0] - R[0,0])
    
    print '\nThis is a fast scan of:', fast_axis
    print 'From:', R[0,1], 'to:', R[-1,1], 'steps:', steps[1], 'range:', np.abs(R[-1,1] - R[0,1])

    # Fourier relationship
    dx   = 1. / (np.array(mask.shape) * dq)
    
    print '\nFourier relation: Real-space'
    print 'dx               :', dx, 'm'
    print 'X (field of view):', dx * np.array(mask.shape), 'm'
    
    print '\nFresnel relation: Sample-plane'
    dxs = du * params['input']['defocus'] / z
    F   = np.array(mask.shape)**2 * dx**2 / (lamb * params['input']['defocus'])
    F   = F[0]
    print 'dx               :', dxs, 'm'
    print 'X (field of view):', dxs * np.array(mask.shape), 'm'
    print 'Fresnel number   :', F
    
    dx = dxs
     
    Rpix = np.zeros( (len(Rindex), 3), dtype=np.int)
    Rpix[:, 0] = np.rint(R[:, 0] / dx[0]).astype(np.int)
    Rpix[:, 1] = np.rint(R[:, 1] / dx[1]).astype(np.int)
    Rpix[:, 2] = np.rint(z / dx[0]).astype(np.int)
        
    print '\nConverting X Y displacements of the sample into pixel units:'
    print '\nSlow scan:'
    print 'range:', Rpix[:, 0].max() - Rpix[:, 0].min(), 'pixels', \
          'average step size:', (Rpix[:, 0].max() - Rpix[:, 0].min())/float(steps[0])
    
    print '\nFast scan:'
    print 'range:', Rpix[:, 1].max() - Rpix[:, 1].min(), 'pixels', \
          'average step size:', (Rpix[:, 1].max() - Rpix[:, 1].min())/float(steps[1])
            
    return R, Rpix, Rindex, F


def get_Is(f, mask, Rindex, I_crop_pad_downsample, params):
    I = np.zeros( (len(Rindex),) + mask.shape, dtype=np.float)
    
    for i, step in enumerate(Rindex) :
        t    = f['entry_1/data_1/data'][step]
        I[i] = I_crop_pad_downsample(t)

    print '\nprocessing frames:',I.shape, I.dtype
    print '\nprocessing steps:',Rindex

    if params['input']['minimum_photons'] is not None :
        I       -= params['input']['minimum_photons'] 
        t0, t1   = I.size, np.sum(I<0)
        I[I < 0] = 0
        print 'thresholding intensities below', params['input']['minimum_photons'], 'photons'
        print t1, 'out of', t0, 'pixels {0:.3f}'.format( 100 * t1 / t0 )

    return I

def make_P0(f, mask, Rindex, F, I_crop_pad_downsample, params):
    """
    Make an initial guess for the probe in 'sample' space by summing the white field.
    """
    # whitefield just sum for now
    print '\nGenerating whitefield (just the mean of the diffraction patterns)'
    #whitefield = np.sqrt(np.mean(I, axis=0)) * mask
    whitefield = np.sqrt(f['process_2/powder'].value / float(f['entry_1/data_1/data'].shape[0]))
    whitefield = I_crop_pad_downsample(whitefield)
    
    # fill in the zeros with the mean value of the whitefield
    print 'filling bad pixels with neighbouring values'
    good_vals = np.where(mask.ravel())[0]
    bad_vals  = np.where(~mask.ravel())[0]
    flat      = whitefield.ravel()
    for i in bad_vals :
        # find the closest good value 
        j = np.argmin(np.abs(i - good_vals))
        j = good_vals[j]
        
        good_2d = np.unravel_index(j, whitefield.shape)
        bad_2d  = np.unravel_index(i, whitefield.shape)
        whitefield[bad_2d] = whitefield[good_2d]
        
        print i, '-->', j, bad_2d , '-->', good_2d, whitefield[bad_2d], '-->', whitefield[good_2d]
    
    print 'ifft-shifting'
    whitefield = np.fft.ifftshift(whitefield)
    
    # defocus
    print 'making the phase factor'
    shape = whitefield.shape
    i     = np.fft.fftfreq(shape[0], 1/float(shape[0]))
    j     = np.fft.fftfreq(shape[1], 1/float(shape[1]))
    i, j  = np.meshgrid(i, j, indexing='ij')
    
    # apply phase
    exps = np.exp(1.0J * np.pi * (i**2 * F / shape[0]**2 + \
                                  j**2 * F / shape[1]**2))
    def prop(x):
        out = np.fft.ifftn(np.fft.ifftshift(x, axes=(-2,-1)), axes = (-2, -1)) * exps.conj() 
        out = np.fft.fftn(out, axes = (-2, -1))
        return out
    
    def iprop(x):
        out = np.fft.ifftn(x, axes = (-2, -1)) * exps
        out = np.fft.fftn(out, axes = (-2, -1))
        return np.fft.ifftshift(out, axes=(-2,-1))
    
    print 'propagating the whitefield to the sample plane'
    P0 = iprop(whitefield + 0J)
    #P0 = np.fft.fftshift(whitefield) + 0J
    return P0, prop, iprop, np.fft.fftshift(whitefield)

def make_O0(I, R, P, O, mask):
    method, update, dtype, c_dtype, MPI_dtype, MPI_c_dtype, OP_iters, O, P, amp, Pamp, background, R, mask, I_norm, N, exits = \
            era.preamble(I, R, P, O, None, None, mask, None, 3, 'mpi', 1.0e-10, 'double', False)
    
    O.fill(1.)
    
    # set the exit waves to the sqrt(I)
    exits = np.fft.fftshift(amp, axes=(-2,-1)) + 0J  
    
    P_heatmap    = None
    O_heatmap    = None
    O, P_heatmap = era.psup_O(exits, P, R, O.shape, P_heatmap, 1.0e-10, MPI_dtype, MPI_c_dtype, False, None)
    return O

if __name__ == '__main__':
    params = utils.parse_cmdline_args()
    
    if rank == 0 :
        print '\n\nLoading', params['input']['cxi_fnam'] 
        f = h5py.File(params['input']['cxi_fnam'], 'r')
         
        print '\n\nMask'
        print '####'
        mask, I_crop_pad_downsample = get_mask(f, params)
                
        print '\n\nR'
        print '####'
        R, Rpix, Rindex, F = get_Rs(f, mask, params)
        
        print '\n\nI'
        print '####'
        I = get_Is(f, mask, Rindex, I_crop_pad_downsample, params)
        
        print '\n\nP0'
        print '####'
        P0, prop, iprop, whitefield = make_P0(f, mask, Rindex, F, I_crop_pad_downsample, params)
        
        print '\n\nO0'
        print '####'
        
        #O0 = make_O0(I, Rpix, whitefield + 0J, None, mask)
        O0 = make_O0(I, Rpix, P0, None, mask)
        """
        Os = []
        heatmaps = []
        for defocus in np.linspace(1.0e-3, 3.0e-3, 50):
            print '\n\nDefocus:', defocus
            params['input']['defocus'] = defocus
            R, Rpix, Rindex, F = get_Rs(f, mask, params)
            
            O0, P_heatmap = make_O0(I, Rpix, whitefield + 0J, None, mask)
            Os.append(O0.copy())
            heatmaps.append(P_heatmap.copy())
        """
        
        print '\n Array shapes:'
        print '\t I   ', I.shape
        print '\t P0  ', P0.shape
        print '\t mask', mask.shape
        print '\t R   ', R.shape



     
