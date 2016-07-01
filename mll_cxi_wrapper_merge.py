"""
Load an mll cxi file with:
process_2/dark
process_2/whitefield
process_2/background
process_2/ptycho_mask
"""

import numpy as np
import h5py
import scipy.constants as sc
import time
from scipy import ndimage
import ConfigParser
import sys, os

import Ptychography.ptychography.era as era
from Ptychography import DM
from Ptychography import ERA
from Ptychography import utils
from Ptychography import write_cxi

from mll_cxi_wrapper import *

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def parse_cmdline_args():
    import argparse
    parser = argparse.ArgumentParser(prog = 'mpirun -n N python mll_cxi_wrapper.py', description='')
    parser.add_argument('config', type=str, \
                        help="configuration file name")

    args = parser.parse_args()

    config_files = args.config.split(',')
    paramss = []

    for c in config_files :
        # check that args.config exists
        if not os.path.exists(c):
            raise NameError('config file does not exist: ' + c)

        # process config file
        config = ConfigParser.ConfigParser()
        config.read(c)
        
        paramss.append(utils.parse_parameters(config))
    return paramss


if __name__ == '__main__':
    paramss = parse_cmdline_args()

    if rank == 0 :
        for i, params in enumerate(paramss):
            print '\n\nLoading', params['input']['cxi_fnam'] 
            f = h5py.File(params['input']['cxi_fnam'], 'r')
             
            if i == 0 :
                print '\n\nMask'
                print '####'
                mask, I_crop_pad_downsample, crop_pad_downsample_nomask = get_mask(f, params)
                    
            print '\n\nR'
            print '####'
            Ri, Rpixi, Rindexi, Fi = get_Rs(f, mask, params)
            print np.diff(Rpixi[:,0])
            print np.diff(Rpixi[:,1])
            #sys.exit()

            print '\n\nI'
            print '####'
            Ii = get_Is(f, mask, Rindexi, I_crop_pad_downsample, params)
        
            print '\n\nP0'
            print '####'
            P0i, prop, iprop, whitefieldi, exps = make_P0(f, mask, Rindexi, Fi, I_crop_pad_downsample, crop_pad_downsample_nomask, params)

            scale = np.mean(f['entry_1/instrument_1/detector_1/count_time'].value)
            Ii          /= scale
            P0i         /= np.sqrt(scale)
            whitefieldi /= scale
            if i == 0 :
                Rpix = Rpixi.copy()
                I    = Ii.copy()
                F    = Fi
                P0   = P0i.copy() 
                whitefield = whitefieldi.copy() 
            else :
                Rpix = np.vstack((Rpix, Rpixi))
                I    = np.vstack((I, Ii))
                F    = Fi
                P0   += P0i
                whitefield += whitefieldi.copy()
                
        P0         /= float(len(paramss))
        whitefield /= float(len(paramss))
        
        #print '\n\nO0'
        #print '####'
        #O = back_prop(prop, iprop, I, whitefield, f, mask, Rpix, params)
        
        print '\n Array shapes:'
        print '\t I   ', I.shape
        print '\t P0  ', P0.shape
        print '\t mask', mask.shape
        print '\t R   ', Rpix.shape
     
    comm.Barrier() 
    params = paramss[0]

    # Initialise
    ############
    eMod = []
    if rank == 0 :
        P  = P0.copy()
        #
        info = {}
        info['I'] = I
        P00 = whitefield + 0J
    else :
        F = I = Rpix = P = O = mask = P00 = whitefield = f = None


    if rank == 0 : 
        print '\n\nO0'
        print '####'
    O0 = back_prop(I, P, whitefield, f, mask, Rpix, F, params)
    O  = O0.copy()
    if rank == 0 : print '\t O   ', O.shape


    # Phase
    ############
    if rank == 0 : print '\n\nPhase'
    if rank == 0 : print '#####'
    
    d0 = time.time()

    if params['phasing']['fresnel_psup'] :
        O, P = Psup(I, Rpix, P, O, params['phasing']['era_iters'], OP_iters = params['phasing']['op_iters'], \
                          mask = mask, background = None, method = params['phasing']['method'], 
                          hardware = 'cpu', alpha = params['phasing']['alpha'], dtype = params['phasing']['dtype'], full_output = True, \
                          verbose = False, sample_blur = params['phasing']['sample_blur'])
    
    for i in range(params['phasing']['outer_loop']): 
        if params['phasing']['dm_iters'] > 0 :
            O, P, info =  DM(I, Rpix, P, O, params['phasing']['dm_iters'], OP_iters = params['phasing']['op_iters'], \
                          mask = mask, background = None, method = params['phasing']['method'], Pmod_probe = params['phasing']['pmod_probe'] , \
                          probe_centering = params['phasing']['probe_centering'], hardware = 'cpu', \
                          alpha = params['phasing']['alpha'], dtype = params['phasing']['dtype'], full_output = True)
                          #sample_blur = params['phasing']['sample_blur'])

            if rank == 0 : eMod += info['eMod']

        if params['phasing']['era_iters'] > 0 :
            O, P, info =  ERA(I, Rpix, P, O, params['phasing']['era_iters'], OP_iters = params['phasing']['op_iters'], \
                          mask = mask, Fresnel = F, background = None, method = params['phasing']['method'], Pmod_probe = params['phasing']['pmod_probe'] , \
                          probe_centering = params['phasing']['probe_centering'], hardware = 'cpu', \
                          alpha = params['phasing']['alpha'], dtype = params['phasing']['dtype'], full_output = True, verbose = False, \
                          sample_blur = params['phasing']['sample_blur'])

            if rank == 0 : eMod += info['eMod']
    
    d1 = time.time()
    
    # Output
    ############
    if rank == 0 :
        print '\ntime:', d1-d0
        write_cxi(I, info['I'], P0, P, O0, O, \
                  Rpix, Rpix, None, None, mask, eMod, fnam = params['output']['fnam'], compress = True)
