"""
scan.cxi --> Ptychography

Get a pupil function

Do a few iterations on the gpu
    see if the run fits on the gpu (use float32)
    I can also truncate the data probably

"""

import numpy as np
#import pyqtgraph as pg

import ptychography as pt
from ptychography.Ptychography_2dsample_2dprobe_farfield import forward_sim

print '\nMaking the forward simulated data...'
I, R, mask, P, O, sample_support = forward_sim()

print '\nUpdating the object on a single cpu core...'
Or, info = pt.ERA(I, R, P, None, 10, mask=mask, alpha=1e-10, dtype='double')

print '\nUpdating the probe on a single cpu core...'
Pr, info = pt.ERA(I, R, None, O, 10, mask=mask, alpha=1e-10)

print '\nUpdating the object and probe on a single cpu core...'
Or, Pr, info = pt.ERA(I, R, None, None, 10, mask=mask, alpha=1e-10)

print '\nUpdating the object a single gpu...'
Or, info = pt.ERA(I, R, P, None, 10, hardware = 'gpu', mask=mask, alpha=1e-10, dtype='double')

print '\nUpdating the probe a single gpu...'
Pr, info = pt.ERA(I, R, None, O, 10, hardware = 'gpu', mask=mask, alpha=1e-10, dtype='double')

print '\nUpdating the object and probe a single gpu...'
Or, Pr, info = pt.ERA(I, R, None, None, 10, hardware = 'gpu', method = 6, mask=mask, alpha=1e-10)

"""
back = (np.random.random(I[0].shape) * 10.0 * np.mean(I))**2
I   += back

print '\n\nWithout background retrieval:'
Or1, Pr1, info = pt.ERA(I, R, None, None, 1000, mask=mask, method = 3, alpha=1e-10, dtype='double')

print '\n\nWith background retrieval:'
Or2, Pr2, background, info = pt.ERA(I, R, None, None, 1000, mask=mask, method = 9, alpha=1e-10, dtype='double')
"""


