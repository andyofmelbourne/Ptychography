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

from mll_cxi_wrapper import *

if __name__ == '__main__':
    params = utils.parse_cmdline_args()
    
    if rank == 0 :
        print '\n\nLoading', params['output']['fnam'] 
        g = h5py.File(params['output']['fnam'], 'r')
        f = h5py.File(params['input']['cxi_fnam'], 'r')
         
        print '\n\nMask'
        print '####'
        mask, I_crop_pad_downsample, crop_pad_downsample_nomask = get_mask(f, params)
                
        print '\n\nR'
        print '####'
        R, Rpix, Rindex, F = get_Rs(f, mask, params)
        
        print '\n\nP'
        print '####'
        P0, prop, iprop, whitefield, exps = make_P0(f, mask, Rindex, F, I_crop_pad_downsample, crop_pad_downsample_nomask, params)

        P = np.fft.fftshift(prop(g['output']['P'].value))
        #I = get_Is(f, mask, Rindex, I_crop_pad_downsample, params)

        #O0 = make_O0(I, Rpix, whitefield + 0J, None, mask)
        #O0 = make_O0(I, Rpix, P0, None, mask)
        
        import pyqtgraph as pg

     
