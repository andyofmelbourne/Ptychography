"""
scan.cxi --> Ptychography

Get a pupil function

Do a few iterations on the gpu
    see if the run fits on the gpu (use float32)
    I can also truncate the data probably

"""

import numpy as np
import time
#import pyqtgraph as pg

import ptychography as pt
from ptychography.Ptychography_2dsample_2dprobe_farfield import forward_sim

try :
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except Exception as e:
    rank = 0
    size = 1

if rank == 0 :
    try :
        print '\nMaking the forward simulated data...'
        I, R, mask, P, O, sample_support = forward_sim()
    except Exception as e:
        print e

    """
    try :
        d0 = time.time()
        print '\nUpdating the object on a single cpu core...'
        Or, info = pt.ERA(I, R, P, None, 100, mask=mask, alpha=1e-10, dtype='double')
        d1 = time.time()
        print '\ntime (s):', (d1 - d0) 
    except Exception as e:
        print e
    """

    try :
        d0 = time.time()
        print '\nUpdating the probe on a single cpu core...'
        Pr, info = pt.ERA(I, R, None, O, 100, mask=mask, alpha=1e-10)
        d1 = time.time()
        print '\ntime (s):', (d1 - d0) 
    except Exception as e:
        print e

"""
    try :
        print '\nUpdating the object and probe on a single cpu core...'
        Or, Pr, info = pt.ERA(I, R, None, None, 10, mask=mask, alpha=1e-10)
    except Exception as e:
        print e

    try :
        print '\nUpdating the object a single gpu...'
        Or, info = pt.ERA(I, R, P, None, 10, hardware = 'gpu', mask=mask, alpha=1e-10, dtype='double')
    except Exception as e:
        print e

    try :
        print '\nUpdating the probe a single gpu...'
        Pr, info = pt.ERA(I, R, None, O, 10, hardware = 'gpu', mask=mask, alpha=1e-10, dtype='double')
    except Exception as e:
        print e

    try :
        print '\nUpdating the object and probe a single gpu...'
        Or, Pr, info = pt.ERA(I, R, None, None, 10, hardware = 'gpu', method = 6, mask=mask, alpha=1e-10)
    except Exception as e:
        print e

    back = (np.random.random(I[0].shape) * 10.0 * np.mean(I))**2
    I   += back

    try :
        print '\nUpdating the object on a single cpu core with background retrieval...'
        Or2, background, info = pt.ERA(I, R, P, None, 10, mask=mask, method = 4, alpha=1e-10, dtype='double')
    except Exception as e:
        print e

    try :
        print '\nUpdating the probe on a single cpu core with background retrieval...'
        Pr2, background, info = pt.ERA(I, R, None, O, 10, mask=mask, method = 5, alpha=1e-10, dtype='double')
    except Exception as e:
        print e

    try :
        print '\nUpdating the object and probe on a single cpu core with background retrieval...'
        Or2, Pr2, background, info = pt.ERA(I, R, None, None, 10, mask=mask, method = 6, alpha=1e-10, dtype='double')
    except Exception as e:
        print e
"""

if rank != 0 :
    I = R = O = P = mask = None

"""
try :
    d0 = time.time()
    if rank == 0 : print '\nUpdating the object on a many cpu cores...'
    Or2, info = pt.ERA(I, R, P, None, 100, hardware = 'mpi', mask=mask, method = 1, alpha=1e-10, dtype='double')
    d1 = time.time()
    if rank == 0 : print '\ntime (s):', (d1 - d0) 
except Exception as e:
    print e
"""

d0 = time.time()
if rank == 0 : print '\nUpdating the probe on a many cpu cores...'
Or2, info = pt.ERA(I, R, None, O, 100, hardware = 'mpi', mask=mask, method = 2, alpha=1e-10, dtype='double')
d1 = time.time()
if rank == 0 : print '\ntime (s):', (d1 - d0) 

"""
try :
    d0 = time.time()
    if rank == 0 : print '\nUpdating the probe on a many cpu cores...'
    Or2, info = pt.ERA(I, R, None, O, 100, hardware = 'mpi', mask=mask, method = 2, alpha=1e-10, dtype='double')
    d1 = time.time()
    if rank == 0 : print '\ntime (s):', (d1 - d0) 
except Exception as e:
    print e
"""

if rank == 0 : print '\n\nDone!'


