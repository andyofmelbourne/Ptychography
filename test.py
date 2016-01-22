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

#Or, info = pt.ERA(I, R, P, None, 100, mask=mask, alpha=1e-10)

#Pr, info = pt.ERA(I, R, None, O, 100, update = 'P', mask=mask, alpha=1e-10)

#Or, Pr, info = pt.ERA(I, R, None, None, 1000, update = 'OP', mask=mask, alpha=1e-10)

Or, info = pt.ERA(I, R, P, O, 10, method = 4, mask=mask, alpha=1e-10)
