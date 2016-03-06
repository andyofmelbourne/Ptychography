import numpy as np
import time
import sys

from ptychography.forward_models import forward_sim
from ptychography.display import write_cxi
from ptychography.dm_mpi import DM_mpi
from ptychography.era import ERA

"""
# test cpu
#---------
print '\nMaking the forward simulated data...'
I, R, M, P, O, B = forward_sim(shape_P = (128, 128), shape_O = (256, 256), A = 32, defocus = 1.0e-2,\
                  photons_pupil = 1, ny = 10, nx = 10, random_offset = 5, \
                  background = None, mask = 100, counting_statistics = False)
# make the masked pixels bad
I += 10000. * ~M 

# initial guess for the probe 
P0 = np.fft.fftshift( np.fft.ifftn( np.abs(np.fft.fftn(P)) ) )

print '\n-------------------------------------------'
print 'Updating the probe with the farfield modulus kept constant.'
d0 = time.time()

Or, Pr, info = ERA(I, R, P0, None, 100, mask=M, method = 3, hardware = 'cpu', Pmod_probe = False, alpha=1e-10, dtype='double')

d1 = time.time()
print '\ntime (s):', (d1 - d0) 

write_cxi(I, info['I'], P, Pr, O, Or, \
      R, None, None, None, M, info['eMod'], fnam = 'output_method3.cxi')

# test mpi
#---------
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0 :
    print '\nMaking the forward simulated data...'
    I, R, M, P, O, B = forward_sim(shape_P = (128, 128), shape_O = (256, 256), A = 32, defocus = 1.0e-2,\
                      photons_pupil = 1, ny = 10, nx = 10, random_offset = 5, \
                      background = None, mask = 100, counting_statistics = False)
    # make the masked pixels bad
    I += 10000. * ~M 

    # initial guess for the probe 
    P0 = np.fft.fftshift( np.fft.ifftn( np.abs(np.fft.fftn(P)) ) )

    print '\n-------------------------------------------'
    print 'Updating the probe with the farfield modulus kept constant.'
    d0 = time.time()
else :
    I = R = P0 = O = M = None

Or, Pr, info = ERA(I, R, P0, O, 1000, mask=M, method = 3, hardware = 'mpi', Pmod_probe = True, alpha=1e-10, dtype='double')

if rank == 0 :
    d1 = time.time()
    print '\ntime (s):', (d1 - d0) 

    write_cxi(I, info['I'], P, Pr, O, O, \
          R, None, None, None, M, info['eMod'], fnam = 'output_method3.cxi')
"""
