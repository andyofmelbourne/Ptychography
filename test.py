"""
scan.cxi --> Ptychography

Get a pupil function

Do a few iterations on the gpu
    see if the run fits on the gpu (use float32)
    I can also truncate the data probably

"""

import numpy as np
import time
import sys
#import pyqtgraph as pg

import ptychography as pt
from ptychography.Ptychography_2dsample_2dprobe_farfield import forward_sim


if len(sys.argv) == 2 :
    iters = int(sys.argv[1])
    test = 'all'
elif len(sys.argv) == 3 :
    iters = int(sys.argv[1])
    test = sys.argv[2]
else :
    iters = 10
    test = 'all'

if test == 'all' or test == 'mpi':
    try :
        from mpi4py import MPI
    
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except Exception as e:
        print e
else: 
    rank = 0
    size = 1

if rank == 0 :
    try :
        print '\nMaking the forward simulated data...'
        I, R, mask, P, O, sample_support = forward_sim()
    except Exception as e:
        print e

    if test == 'all' or test == 'cpu':
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


    if test == 'all' or test == 'gpu':
        # Single gpu core 
        #----------------
        print '\n-----------------------------------'
        print 'Updating the object a single gpu...'
        try :
            d0 = time.time()
            Or, info = pt.ERA(I, R, P, None, iters, hardware = 'gpu', method=1, mask=mask, alpha=1e-10, dtype='double')
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
        except Exception as e:
            print e
    
        print '\nUpdating the probe a single gpu...'
        try :
            d0 = time.time()
            Pr, info = pt.ERA(I, R, None, O, iters, hardware = 'gpu', method=2, mask=mask, alpha=1e-10, dtype='double')
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
        except Exception as e:
            print e
    
        print '\nUpdating the object and probe a single gpu...'
        try :
            d0 = time.time()
            Or, Pr, info = pt.ERA(I, R, None, None, iters, hardware = 'gpu', method = 3, mask=mask, alpha=1e-10)
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
        except Exception as e:
            print e

if test == 'all' or test == 'mpi':
    # Many cpu cores 
    #----------------
    if rank != 0 :
        I = R = O = P = mask = None
    
    if rank == 0 : print '\n------------------------------------------'
    if rank == 0 : print 'Updating the object on a many cpu cores...'
    try :
        d0 = time.time()
        Or2, info = pt.ERA(I, R, P, None, iters, hardware = 'mpi', mask=mask, method = 1, alpha=1e-10, dtype='double')
        d1 = time.time()
        if rank == 0 : print '\ntime (s):', (d1 - d0) 
    except Exception as e:
        print e
    
    if rank == 0 : print '\nUpdating the probe on a many cpu cores...'
    try :
        d0 = time.time()
        Or2, info = pt.ERA(I, R, None, O, iters, hardware = 'mpi', mask=mask, method = 2, alpha=1e-10, dtype='double')
        d1 = time.time()
        if rank == 0 : print '\ntime (s):', (d1 - d0) 
    except Exception as e:
        print e
    
    if rank == 0 : print '\nUpdating the object and probe on a many cpu cores...'
    d0 = time.time()
    Or2, Pr2, info = pt.ERA(I, R, None, None, iters, hardware = 'mpi', mask=mask, method = 3, alpha=1e-10, dtype='double')
    d1 = time.time()
    if rank == 0 : print '\ntime (s):', (d1 - d0) 
    try :
        pass
    except Exception as e:
        print e


# Single cpu core with background 
#--------------------------------
if rank == 0 :
    if test == 'all' or test == 'cpu':
        back = (np.random.random(I[0].shape) * 10.0 * np.mean(I))**2
        I   += back

        print '\n---------------------------------------------------------------------'
        print 'Updating the object on a single cpu core with background retrieval...'
        try :
            d0 = time.time()
            Or2, background, info = pt.ERA(I, R, P, None, iters, mask=mask, method = 4, alpha=1e-10, dtype='double')
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
        except Exception as e:
            print e

        print '\nUpdating the probe on a single cpu core with background retrieval...'
        try :
            d0 = time.time()
            Pr2, background, info = pt.ERA(I, R, None, O, iters, mask=mask, method = 5, alpha=1e-10, dtype='double')
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
        except Exception as e:
            print e

        print '\nUpdating the object and probe on a single cpu core with background retrieval...'
        try :
            d0 = time.time()
            Or2, Pr2, background, info = pt.ERA(I, R, None, None, iters, mask=mask, method = 6, alpha=1e-10, dtype='double')
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
        except Exception as e:
            print e

    if test == 'all' or test == 'gpu':
        # Single gpu core with background
        #--------------------------------
        print '\n---------------------------------------------------'
        print 'Updating the object a single gpu with background...'
        try :
            d0 = time.time()
            Or, info = pt.ERA(I, R, P, None, iters, hardware = 'gpu', method = 4, mask=mask, alpha=1e-10, dtype='double')
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
        except Exception as e:
            print e
    
        print '\nUpdating the probe a single gpu with background...'
        try :
            d0 = time.time()
            Pr, info = pt.ERA(I, R, None, O, iters, hardware = 'gpu', method = 5, mask=mask, alpha=1e-10, dtype='double')
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
        except Exception as e:
            print e
        
        print '\nUpdating the object and probe a single gpu with background...'
        try :
            d0 = time.time()
            Or, Pr, info = pt.ERA(I, R, None, None, iters, hardware = 'gpu', method = 6, mask=mask, alpha=1e-10)
            d1 = time.time()
            print '\ntime (s):', (d1 - d0) 
        except Exception as e:
            print e

if test == 'all' or test == 'mpi':
    # Many cpu cores with background 
    #-------------------------------
    
    if rank == 0 : print '\n----------------------------------------------------------'
    if rank == 0 : print 'Updating the object with background on a many cpu cores...'
    try :
        d0 = time.time()
        Or2, background, info = pt.ERA(I, R, P, None, iters, hardware = 'mpi', mask=mask, method = 4, alpha=1e-10, dtype='double')
        d1 = time.time()
        if rank == 0 : print '\ntime (s):', (d1 - d0) 
    except Exception as e:
        print e
    
    if rank == 0 : print '\nUpdating the probe with background on a many cpu cores...'
    try :
        d0 = time.time()
        Or2, background, info = pt.ERA(I, R, None, O, iters, hardware = 'mpi', mask=mask, method = 5, alpha=1e-10, dtype='double')
        d1 = time.time()
        if rank == 0 : print '\ntime (s):', (d1 - d0) 
    except Exception as e:
        print e
    
    if rank == 0 : print '\nUpdating the object and probe with background on a many cpu cores...'
    try :
        d0 = time.time()
        Or2, Pr2, background, info = pt.ERA(I, R, None, None, iters, OP_iters = 1, hardware = 'mpi', mask=mask, method = 6, alpha=1e-10, dtype='double')
        d1 = time.time()
        if rank == 0 : print '\ntime (s):', (d1 - d0) 
    except Exception as e:
        print e


if rank == 0 : print '\n\nDone!'


