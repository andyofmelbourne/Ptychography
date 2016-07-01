import numpy as np
import h5py
import sys

def if_exists_del(fnam):
    import os
    # check that the directory exists and is a directory
    output_dir = os.path.split( os.path.realpath(fnam) )[0]
    if os.path.exists(output_dir) == False :
        raise ValueError('specified path does not exist: ', output_dir)
    
    if os.path.isdir(output_dir) == False :
        raise ValueError('specified path is not a path you dummy: ', output_dir)
    
    # see if it exists and if so delete it 
    # (probably dangerous but otherwise this gets really anoying for debuging)
    if os.path.exists(fnam):
        print '\n', fnam ,'file already exists, deleting the old one and making a new one'
        os.remove(fnam)

def write_cxi(I_in, I_out, P_in, P_out, O_in, O_out, \
              R_in, R_out, B_in, B_out, M, eMod, fnam = 'output.cxi', compress = True):
    """
    Write a psudo cxi file for loading and displaying later.
    
    Warning: this will overwrite any existing file.
    """
    if_exists_del(fnam)
    f = h5py.File(fnam, 'w')
    gin = f.create_group('input')
    if compress :
        gin.create_dataset('I', data = I_in, compression = 'gzip')
    else :
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
    if compress :
        gout.create_dataset('I', data = I_out, compression = 'gzip')
    else :
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


def read_cxi(fnam = 'output.cxi', maxlen=100):
    """
    Read a psudo cxi file for loading and displaying later.
    """
    if type(fnam) == str :
        f = h5py.File(f, 'r')
    else :
        f = fnam
    
    gin  = f['input']
    I_in = gin['I'][:maxlen]
    M    = gin['M'].value
    keys = gin.keys()
    if 'P' in keys:
        P_in = gin['P'].value
    else :
        P_in = None
    if 'O' in keys:
        O_in = gin['O'].value
    else :
        O_in = None
    if 'R' in keys:
        R_in = gin['R'].value
    else :
        R_in = None
    if 'B' in keys:
        B_in = gin['B'].value
    else :
        B_in = None

    gout  = f['output']
    I_out = gout['I'][:maxlen]
    keys  = gout.keys()
    if 'P' in keys:
        P_out = gout['P'].value
    else :
        P_out = None
    if 'O' in keys:
        O_out = gout['O'].value
    else :
        O_out = None
    if 'R' in keys:
        R_out = gout['R'].value
    else :
        R_out = None
    if 'B' in keys:
        B_out = gout['B'].value
    else :
        B_out = None
    if 'eMod' in keys:
        eMod  = gout['eMod'].value
    else :
        eMod = None
    f.close()
    return I_in, I_out, P_in, P_out, O_in, O_out, R_in, R_out, B_in, B_out, M, eMod

def hstack_if_not_None(A, B):
    if (A is not None) and (B is not None) :
        C = np.hstack((A, B))
    elif (A is not None):
        C = A
    elif (B is not None):
        C = B
    return C

def in_vs_out_widget(A, B, f = None, title = ''):
    if (A is None) and (B is None) :
        return None
    import pyqtgraph as pg
    A_in_out_plots = pg.image(title = title)

    print title
    A_in_out = hstack_if_not_None(A, B)

    if f is not None :
        A_in_out = f(A_in_out)
    
    if len(A_in_out.shape) == 2 :
        A_in_out_plots.setImage(A_in_out.T)
    elif len(A_in_out.shape) == 3 :
        A_in_out_plots.setImage(A_in_out)
    
    return A_in_out_plots

def Application(I_in, I_out, P_in, P_out, O_in,  \
                O_out, R_in, R_out, B_in, B_out, \
                M, eMod, maxlen = 100):

    # start a pyqtgraph application (sigh...)
    import pyqtgraph as pg
    from PyQt4 import QtGui, QtCore
    import signal
    
    # Always start by initializing Qt (only once per application)
    app = QtGui.QApplication([])

    wI  = in_vs_out_widget(M*I_in[:maxlen]**0.5, I_out[:maxlen]**0.5, title = 'input / output diffraction intensities')
    wP  = in_vs_out_widget(P_in, P_out, np.abs, 'input / output |Probe|')

    P_in  = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(P_in )))
    P_out = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(P_out)))
    wPa   = in_vs_out_widget(P_in, P_out, np.abs, 'input / output farfield |Probe|')
    wPp   = in_vs_out_widget(P_in, P_out, np.angle, 'input / output farfield angle(Probe)')

    # check if O_out is smaller than O_in
    if (O_in is not None and O_out is not None) and (O_out.shape[0] < O_in.shape[0]) :
        O_out.resize((O_in.shape[0], O_out.shape[1]), refcheck=False)
    wOa = in_vs_out_widget(O_in, O_out, np.abs, 'input / output |Object|')
    wOp = in_vs_out_widget(O_in, O_out, np.angle, 'input / output angle(Object)')
    try :
        wB  = in_vs_out_widget(B_in, B_out, title = 'input / output background')
    except :
        pass
    wM  = in_vs_out_widget(M, None, title = 'detector mask')

    eMod_plot = pg.plot(eMod, title = 'Modulus error')
    eMod_plot.setLabel('bottom', 'iteration')
    eMod_plot.setLabel('left', 'error')

    R_plot = pg.plot(R_in[:, 0], title = 'sample position ss, fs (red, green)', pen=pg.mkPen('r'))
    R_plot.plot(R_in[:, 1], pen=pg.mkPen('g'))
    R_plot.setLabel('bottom', 'index')
    R_plot.setLabel('left', 'pixels')
    
    ## Start the Qt event loop
    signal.signal(signal.SIGINT, signal.SIG_DFL)    # allow Control-C
    sys.exit(app.exec_())


def display_cxi(f, maxlen=100):
    if type(f) == str :
        f = h5py.File(f, 'r')
   
    from PyQt4 import QtGui, QtCore
    
    I_in, I_out, P_in, P_out, O_in, O_out, R_in, R_out, B_in, B_out, M, eMod = read_cxi(f, maxlen=100)
    
    Application(I_in, I_out, P_in, P_out, O_in, O_out, R_in, R_out, B_in, B_out, M, eMod)


if __name__ == '__main__':
    
    fnam = sys.argv[1]
    display_cxi(fnam)


