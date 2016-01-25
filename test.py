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

I, R, mask, P, O, sample_support = forward_sim()

#Or, info = pt.ERA(I, R, P, None, 20, mask=mask, alpha=1e-10, dtype='double')

#Pr, info = pt.ERA(I, R, None, O, 100, update = 'P', mask=mask, alpha=1e-10)

#Or, Pr, info = pt.ERA(I, R, None, None, 1000, update = 'OP', mask=mask, alpha=1e-10)

#Or, info = pt.ERA(I, R, P, None, 1000, method = 4, mask=mask, alpha=1e-10, dtype='double')

#Pr, info = pt.ERA(I, R, None, O, 1000, method = 5, mask=mask, alpha=1e-10, dtype='double')

#Or, Pr, info = pt.ERA(I, R, None, None, 1000, method = 6, mask=mask, alpha=1e-10)

back = (np.random.random(I[0].shape) * 10.0 * np.mean(I))**2
I   += back

print '\n\nWithout background retrieval:'
Or1, Pr1, info = pt.ERA(I, R, None, None, 1000, mask=mask, method = 3, alpha=1e-10, dtype='double')

print '\n\nWith background retrieval:'
Or2, Pr2, background, info = pt.ERA(I, R, None, None, 1000, mask=mask, method = 9, alpha=1e-10, dtype='double')


