import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# time all reduce
array  = np.random.random((100, 100, 100)) + 1J
array2 = np.empty_like(array)

import time
d0 = time.time()
for i in range(100):
    comm.Allreduce([array, MPI.__TypeDict__[array.dtype.char]], [array2, MPI.__TypeDict__[array2.dtype.char]], op=MPI.SUM)
d1 = time.time()

dd0 = time.time()
for i in range(100):
    array3 = comm.allreduce(array, op=MPI.SUM)
dd1 = time.time()

if rank==0 :
    print 'numpy:  delta t:', (d1 - d0) 
    print 'pickle: delta t:', (dd1 - dd0) 
    print array2.max()
    print array3.max()


