import numpy as np
import h5py

def write_cxi(I_in, I_out, P_in, P_out, O_in, O_out, \
              R_in, R_out,B_in, B_out, M, eMod, fnam = 'output.cxi'):
    """
    Write a psudo cxi file for loading and displaying later.
    """
    f = h5py.File(fnam, 'w')
    gin = f.create_group('input')
    gin.create_dataset('I', data = I_in)
    gin.create_dataset('P', data = P_in)
    gin.create_dataset('O', data = O_in)
    gin.create_dataset('R', data = R)
    gin.create_dataset('M', data = B)
