import numpy as np
import scipy.constants
import h5py
import bagOfns as bg
from Ptychography_2dsample_2dprobe_farfield import Ptychography
from utility_Ptych import makeExits3 as makeExits

class Pcxi(h5py.File):
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
    def __init__(self, cxi_fnam):
        """Initialise the Ptychography class from the cxi file
        """
        h5py.File.__init__(self, cxi_fnam, 'a')

    def ERA(self, iters=1, update='sample'):
        """
        self --> Ptychography
        ERA iters
        self <-- Ptychography
        """
        prob = Pcxi.Pcxi_to_P(self)
        prob = Ptychography.ERA(prob, iters=iters, update=update)
        self = Pcxi.append_P(self, prob, 'Pcxi.ERA('+self.filename+', '+str(iters)+', update='+update+')')
        return self

    def Thibault(self, iters=1, update='sample'):
        """
        self --> Ptychography
        Thibault iters
        self <-- Ptychography 
        """
        prob = Pcxi.Pcxi_to_P(self)
        prob = Ptychography.Thibault(prob, iters=iters, update=update)
        self = Pcxi.append_P(self, prob, 'Pcxi.Thibault('+self.filename+', '+str(iters)+', update='+update+')')
        return self

    def heatmap(self):
        """
        self --> Ptychography
        heatmap
        """
        prob = Pcxi.Pcxi_to_P(self)
        heatmap = Ptychography.heatmap(prob)
        return heatmap

    def Pcxi_to_P(self):
        diffs  = np.array(self['entry_1/image_1/data'])
        mask   = np.array(self['entry_1/image_1/instrument_1/detector_1/mask'])
        shape  = diffs[0].shape
        
        # geometry
        z      = self['entry_1/image_1/instrument_1/detector_1/distance'].value
        lamb   = self['entry_1/image_1/instrument_1/source_1/wave length'].value
        du     = np.zeros((2), dtype=np.float64)
        du[0]  = self['entry_1/image_1/instrument_1/detector_1/y_pixel_size'].value
        du[1]  = self['entry_1/image_1/instrument_1/detector_1/x_pixel_size'].value
        dr     = lamb * z / (shape * du)
        xyz    = np.array(self['entry_1/image_1/translation'])
        i = xyz[:,1] / dr[0]
        j = xyz[:,0] / dr[1]
        # HACK !!!!!!!!!!!!!!!!!!
        coords = (np.array(zip(i, j)) - 0.01).astype(dtype=np.int) 
        
        # sample / probe
        sample         = np.array(self['entry_1/sample_1/transmission function'])
        sample_support = np.array(self['entry_1/sample_1/support'])
        probe = np.array(self['entry_1/image_1/instrument_1/source_1/probe'])
        
        prob = Ptychography(diffs, coords, probe, sample, mask, sample_support)

        prob.exits = np.array(self['entry_1/image_1/exits'])
        return prob

    def append_P(self, prob, command):
        """Append a comment and update the sample/probe/coords/exits/errors"""
        self['entry_1/sample_1/transmission function'][:]      = prob.sample
        self['entry_1/image_1/instrument_1/source_1/probe'][:] = prob.probe
        self['entry_1/image_1/exits'][:]                       = prob.exits
        
        eMod = self['entry_1/image_1/process_1/eMod']
        temp = eMod.value
        eMod.resize((eMod.shape[0] + np.array(prob.eMod).shape[0], eMod.shape[1]))
        eMod[:] = np.vstack((temp, np.array(prob.eMod)))

        eSup = self['entry_1/image_1/process_1/eSup']
        temp = eSup.value
        eSup.resize((eSup.shape[0] + np.array(prob.eSup).shape[0], eSup.shape[1]))
        eSup[:] = np.vstack((temp, np.array(prob.eSup)))

        self['entry_1/image_1/process_1/command'][0] += np.string_(command + '\n')
        return self


def forward_sim():
    """ """
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

    # random offset 
    dcoords = (np.random.random(coords.shape) * 3).astype(np.int)
    coords += dcoords
    coords[np.where(coords > 0)] = 0
    coords[:, 0][np.where(coords[:, 0] < shape_illum[0] - shape_sample[0])] = shape_illum[0] - shape_sample[0]
    coords[:, 1][np.where(coords[:, 1] < shape_illum[1] - shape_sample[1])] = shape_illum[1] - shape_sample[1]

    # diffraction patterns
    diffs = makeExits(sample, probe, coords)
    diffs = np.abs(bg.fft2(diffs))**2

    mask = np.ones_like(diffs[0], dtype=np.bool)

    data = {}
    data['dr']   = np.array([1.0e-9, 5.0e-10])                   # real space pixel size
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
    f['entry_1/instrument_1/detector_1'].create_dataset('y_pixel_size', data = data['du'][0])
    f['entry_1/instrument_1/detector_1'].create_dataset('x_pixel_size', data = data['du'][1])
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
    f['entry_1/image_1/process_1'].create_dataset('command', (1, ), maxshape=(None, ), dtype='S1000')
    f['entry_1/image_1/process_1'].create_dataset('eMod', (0, 2), maxshape=(None, 2), dtype=np.float64)
    f['entry_1/image_1/process_1'].create_dataset('eSup', (0, 2), maxshape=(None, 2), dtype=np.float64)
    
    bg.print_h5(f)
    
    fnam = f.filename
    f.close()
    return fnam

if __name__ == '__main__':
    cxi_file = put_into_cxi()

    Pcxi_file = Pcxi(cxi_file)

    sample   = np.array(Pcxi_file['entry_1/sample_1/transmission function'])
    probe    = np.array(Pcxi_file['entry_1/image_1/instrument_1/source_1/probe'])
    Pcxi_file['entry_1/sample_1/transmission function'][:] = np.random.random(sample.shape) + 1J * np.random.random(sample.shape)
    
    prob = Pcxi.Thibault(Pcxi_file, 20, update='sample')
    prob = Pcxi.ERA(Pcxi_file, 50, update='sample')
    
    # check the fidelity inside of the illuminated region:
    sample_ret = np.array(Pcxi_file['entry_1/sample_1/transmission function'])
    probe_sum  = Pcxi.heatmap(Pcxi_file)
    probe_mask = probe_sum > probe_sum.max() * 1.0e-10
    
    print '\nfidelity: ', bg.l2norm(sample * probe_mask, sample_ret * probe_mask)
