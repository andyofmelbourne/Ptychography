import numpy as np
import h5py

def write_cxi(I_in, I_out, P_in, P_out, O_in, O_out, \
              R_in, R_out, B_in, B_out, M, eMod, fnam = 'output.cxi'):
    """
    Write a psudo cxi file for loading and displaying later.
    
    Warning: this will overwrite any existing file.
    """
    f = h5py.File(fnam, 'a')
    gin = f.create_group('input')
    gin.create_dataset('I', data = I_in)
    gin.create_dataset('M', data = M)
    if P_in is not None :
        gin.create_dataset('P', data = P_in)
    if O_in is not None :
        gin.create_dataset('O', data = O_in)
    if R_in is not None :
        gin.create_dataset('R', data = R_in)
    if B_in is not None :
        gin.create_dataset('B', data = B_in)

    gout = f.create_group('output')
    gout.create_dataset('I', data = I_out)
    if P_out is not None :
        gout.create_dataset('P', data = P_out)
    if O_out is not None :
        gout.create_dataset('O', data = O_out)
    if R_out is not None :
        gout.create_dataset('R', data = R_out)
    if B_out is not None :
        gout.create_dataset('B', data = B_out)
    if eMod is not None :
        gout.create_dataset('eMod', data = eMod)
    f.close()


def read_cxi(fnam = 'output.cxi'):
    """
    Read a psudo cxi file for loading and displaying later.
    """
    f = h5py.File(fnam, 'r')
    gin  = f['input']
    I_in = gin['I'].value
    M    = gin['M']
    if gin.has_key('P'):
        P_in = gin['P'].value
    else :
        P_in = None
    if gin.has_key('O'):
        O_in = gin['O'].value
    else :
        O_in = None
    if gin.has_key('R'):
        R_in = gin['R'].value
    else :
        R_in = None
    if gin.has_key('B'):
        B_in = gin['B'].value
    else :
        B_in = None

    gout  = f['output']
    I_out = gout['I'].value
    if gout.has_key('P'):
        P_out = gout['P'].value
    else :
        P_out = None
    if gout.has_key('O'):
        O_out = gout['O'].value
    else :
        O_out = None
    if gout.has_key('R'):
        R_out = gout['R'].value
    else :
        R_out = None
    if gout.has_key('B'):
        B_out = gout['B'].value
    else :
        B_out = None
    if gout.has_key('eMod'):
        eMod  = gout['eMod'].value
    else :
        eMod = None
    f.close()
    return I_in, I_out, P_in, P_out, O_in, O_out, R_in, R_out, B_in, B_out, M, eMod


def display_cxi(f):
    if type(f) == str :
        f = h5py.File(f, 'r')
   
    # start a pyqtgraph application (sigh...)


