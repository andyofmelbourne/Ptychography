import numpy as np
import scipy.constants
import h5py
import bagOfns as bg
from Ptychography_2dsample_2dprobe_farfield import Ptychography
from utility_Ptych import makeExits3 as makeExits

class Pcxi(object):
    """A wrapper class that applies the functionality of the Ptychography class to cxi files.
    
    Most of these variables are not used but must be present.
    The cxi file must have at least the following entries:
    /
        cxi_version                             int
        entry_1
             image_1
                 translation                    (L, 3) np.int
                 data                           (L, N, M) np.float
                 instrument_1 
                     detector_1 
                         distance               float 
                         corner_position        (3) np.float
                         y_pixel_size           float
                         x_pixel_size           float
                     source_1 
                         probe_mask             (N, M) bool
                         energy                 float
                         probe                  (N, M) np.complex128 (???)
             data_1 
                 data                           (L, N, M) np.float
             sample_1 
                 thickness                      float
                 description                    str
                 geometry_1 
                     translation                (L, 3) np.int
                 name                           str
             instrument_1 
                 detector_1 
                     distance                   float 
                     corner_position            (3) np.float
                     y_pixel_size               float
                     x_pixel_size               float
                 source_1 
                     probe_mask                 (N, M) bool
                     energy                     float
                     probe                      (N, M) np.complex128 (???)
    """
    def __init__(self, cxi_file):
        """Initialise the Ptychography class from the cxi file
        """
        self.cxi_file = cxi_file

    def ERA(self, iters=1, update='sample'):
        """
        self --> Ptychography
        ERA iters
        Ptychography <-- self
        """
        prob = Pcxi.Pcxi_to_P(self)
        prob = Ptychography.ERA(prob, iters=iters, update=update)
        self.append_P(prob)
        return self

    def Pcxi_to_P(self):
        diffs  = np.array(self.cxi_file['entry_1/image_1/data'])
        mask   = np.array(self.cxi_file['entry_1/image_1/instrument_1/detector_1/mask'])
        shape  = diffs[0].shape
        
        # geometry
        z      = self.cxi_file['entry_1/image_1/instrument_1/detector_1/distance']
        lamb   = self.cxi_file['entry_1/image_1/instrument_1/source_1/wave length']
        du     = np.array((2), dtype=np.float64)
        du[0]  = self.cxi_file['entry_1/image_1/instrument_1/detector_1/y_pixel_size']
        du[1]  = self.cxi_file['entry_1/image_1/instrument_1/detector_1/x_pixel_size']
        dr     = lamb * z / (shape * du)
        xyz    = np.array(self.cxi_file['translation'])
        i = xyz[1] * dr[0]
        j = xyz[0] * dr[1]
        coords = np.array( zip(i, j), dtype=np.int ) 
        
        # sample / probe
        sample         = np.array(self.cxi_file['entry_1/sample_1/transmission function'])
        sample_support = np.array(self.cxi_file['entry_1/sample_1/support'])
        probe = np.array(self.cxi_file['entry_1/image_1/source_1/probe'])
        
        prob = Ptychography(diffs, coords, probe, sample, mask, sample_support)
        return prob

    def append_P(self, prob)
        pass



def forward_sim():
    # sample
    shape_sample = (80, 180)
    amp          = bg.scale(bg.brog(shape_sample), 0.0, 1.0)
    phase        = bg.scale(bg.twain(shape_sample), -np.pi, np.pi)
    sample       = amp * np.exp(1J * phase)
    sample_support = np.ones_like(sample, dtype=np.bool)

    shape_sample = (128, 256)
    sample         = bg.zero_pad(sample,         shape_sample, fillvalue=1.0)
    sample_support = bg.zero_pad(sample_support, shape_sample)
    
    # probe
    shape_illum = (64, 128)
    probe       = bg.circle_new(shape_illum, radius=0.5, origin='centre') + 0J
        
    # make some sample positions
    xs = range(shape_illum[1] - shape_sample[1], 1, 8)
    ys = range(shape_illum[0] - shape_sample[0], 1, 8)
    xs, ys = np.meshgrid( xs, ys )
    coords = np.array(zip(ys.ravel(), xs.ravel()))

    # diffraction patterns
    diffs = makeExits(sample, probe, coords)
    diffs = np.abs(bg.fft2(diffs))**2

    mask = np.ones_like(diffs[0], dtype=np.bool)

    data = {}
    data['dr']   = np.array([1.0e-9, 1.0e-9])                   # real space pixel size
    data['dq']   = 1. / (np.array(diffs[0].shape) * data['dr']) # q space pixel size
    data['du']   = np.array([10.0e-6, 10.0e-6])                 # detector pixel size
    data['lamb'] = 1.0e-9                                       # wave length
    data['energy'] = scipy.constants.h * scipy.constants.c / data['lamb'] 
    data['z']    = data['du'][0] / (data['lamb'] * data['dq'][0])
    data['diffs']          = diffs
    data['coords']         = coords
    data['mask']           = mask
    data['probe']          = probe
    data['sample']         = sample
    data['sample_support'] = sample_support

    y, x = data['coords'][:, 0], data['coords'][:, 1]
    sample_translations_xyz = np.array( zip(x * data['dr'][1], y * data['dr'][0], np.zeros_like(x)) )
    data['sample_translations_xyz'] = sample_translations_xyz
    return data

def put_into_cxi():
    data = forward_sim()

    # get it in there!
    f = h5py.File('Pcxi_example.cxi', 'a')
    f.create_dataset('cxi_version', data = 140)
    f.create_group('entry_1')

    # sample
    f['entry_1'].create_group('sample_1')
    f['entry_1/sample_1'].create_dataset('name', data = 'de Broglie and Twain')
    f['entry_1/sample_1'].create_dataset('description', data = 'simulated sample')
    f['entry_1/sample_1'].create_dataset('thickness', data = 0.0)
    f['entry_1/sample_1'].create_dataset('transmission function', data = data['sample'])
    f['entry_1/sample_1'].create_dataset('support', data = data['sample_support'])
    f['entry_1/sample_1'].create_group('geometry_1')
    f['entry_1/sample_1/geometry_1'].create_dataset('translation', data = data['sample_translations_xyz'])

    f['entry_1'].create_group('instrument_1')
    f['entry_1/instrument_1'].create_group('detector_1')
    z     = data['z']
    shape = data['diffs'].shape
    f['entry_1/instrument_1/detector_1'].create_dataset('distance', data = z)
    f['entry_1/instrument_1/detector_1'].create_dataset('corner_position', data = np.array([ (shape[1]-3)/2., (shape[0]-3)/2., z]))
    f['entry_1/instrument_1/detector_1'].create_dataset('y_pixel_size', data = data['dr'][0])
    f['entry_1/instrument_1/detector_1'].create_dataset('x_pixel_size', data = data['dr'][1])
    f['entry_1/instrument_1/detector_1'].create_dataset('mask', data = data['mask'])
    f['entry_1/instrument_1'].create_group('source_1')
    f['entry_1/instrument_1/source_1'].create_dataset('probe_mask', data = np.ones_like(data['probe'], dtype=np.bool))
    f['entry_1/instrument_1/source_1'].create_dataset('energy', data = data['energy'])
    f['entry_1/instrument_1/source_1'].create_dataset('wave length', data = data['lamb'])
    f['entry_1/instrument_1/source_1'].create_dataset('probe', data = data['probe'])

    f['entry_1'].create_group('data_1')
    f['entry_1/data_1'].create_dataset('data', data = data['diffs'])
    f['entry_1/data_1/data'].attrs['axes'] = ['translation:y:x']
    f['entry_1/data_1/data'].attrs['is_fft_shifted'] = False

    # Add soft links to image_1
    f['entry_1'].create_group('image_1')
    f['entry_1/image_1/translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')
    f['entry_1/image_1/data'] = h5py.SoftLink('/entry_1/data_1/data')
    f['entry_1/image_1/instrument_1'] = h5py.SoftLink('/entry_1/instrument_1')

    # Processed data
    f['entry_1/image_1'].create_dataset('exits', data = np.zeros_like(data['diffs'], dtype=np.complex128)) 
    f['entry_1/image_1'].create_group('process_1')
    f['entry_1/image_1/process_1'].create_dataset('command', data = [])
    
    bg.print_h5(f)
     
    return f

if __name__ == '__main__':
    cxi_file = put_into_cxi()
    
    Pcxi_file = Pcxi(cxi_file)
    
    Pcxi.ERA(Pcxi_file, iters=10, update='sample')

