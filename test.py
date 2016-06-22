import numpy as np
import time
import sys



from Ptychography import forward_sim
from Ptychography import write_cxi
from Ptychography import DM
from Ptychography import ERA

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
if rank == 0 :
    print '\nMaking the forward simulated data...'
    I, R, M, P, O, B, prop, iprop = forward_sim(shape_P = (128, 128), shape_O = (256, 256), A = 32, defocus = 0.0e-2 + 10.0J,\
				      photons_pupil = 1, ny = 10, nx = 10, random_offset = 5, \
				      background = None, mask = 100, counting_statistics = False)
    # make the masked pixels bad
    I += 10000. * ~M 
    
    # initial guess for the probe 
    #P0 = np.fft.fftshift( np.fft.ifftn( np.abs(np.fft.fftn(P)) * np.exp(2.0J * np.pi * np.random.random(P.shape)) ) )
    P0 = P.copy()
else :
    I = R = O = P = P0 = M = B = None

if rank == 0 : 
    print '\n-------------------------------------------'
    print 'Updating the object on ',size ,' cpu cores...'
    d0 = time.time()

Pr = None
Or = None
if rank == 0 : Pr = P0.copy()

eMod = []
#Or, Pr, info = DM(I, R, P0, None, iters, mask=M, method = 3, hardware = 'mpi', alpha=1e-10, dtype='double', probe_centering=True)
#if rank == 0 : eMod += info['eMod']

Or, Pr, info = ERA(I, R, Pr, Or, iters, Fresnel = 1./10.0, mask=M, method = 3, hardware = 'mpi', alpha=1e-10, dtype='double', probe_centering=False)
if rank == 0 : eMod += info['eMod']

if rank == 0 : 
    d1 = time.time()
    print '\ntime (s):', (d1 - d0) 
    
    write_cxi(I, info['I'], P, Pr, O, Or, \
	      R, None, None, None, M, eMod, fnam = 'output_method1.cxi')
