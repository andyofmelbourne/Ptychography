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
    def __init__(self, cxi_fnam, Pcxi):
        """Initialise the Ptychography class from the cxi file
        
        Add the extra entries on top of cxi_fnam.
        """


    
    def write(self, fnam):
        pass

    def read(fnam):
        print fnam
    
    def check(self):
        # recursively search through the data structure
        def myprint(d):
            for k, v in d.iteritems():
                if isinstance(v, dict):
                    myprint(v)
                elif v == None :
                    print "Warning {0} : {1}".format(k, v)
        myprint(self.root)

class P_to_Pcxi(object):
    """Get everything from a Ptychography object and put it into a Pcxi object""" 
    def __init__(self, prob):
        # Grab everything from prob
        sample_translations_xyz = np.array( zip(np.zeros_like(prob.coords[:, 1]), coords[:, 1], coords[:, 0]) )
        probe                   = prob.probe
        probe_mask              = np.ones_like(prob.probe, dtype=np.bool)
        sample_mask             = prob.sample_support
        diff_mask               = prob.mask
        data                    = prob.diffAmps**2
        is_fft_shifted          = 1
        exits                   = prob.exits
        error_mod  = prob.error_mod
        error_sup  = prob.error_sup
        error_conv = prob.error_conv
        
        # Construct the data structure
        self.root = {}
        
        # sample_1
        geometry_1 = {}
        geometry_1['translation'] = sample_translations_xyz
        
        sample_1 = {}
        sample_1['name'] = 'sample'
        sample_1['geometry_1'] = geometry_1
        sample_1['sample_mask'] = sample_mask
        
        # instrument_1
        detector_1 = {}
        detector_1['distance'] = None
        detector_1['corner_position'] = None
        detector_1['x_pixel_size'] = None
        detector_1['y_pixel_size'] = None
        
        source_1 = {}
        source_1['energy'] = None
        source_1['probe'] = probe
        source_1['probe_mask'] = probe_mask
        
        instrument_1 = {}
        instrument_1['detector_1'] = detector_1
        instrument_1['source_1'] = source_1
        
        # image_1, this stuff get linked to
        image_1 = {}
        image_1['data'] = None
        image_1['translation'] = None
        image_1['intrument_1'] = None
        
        # process_1 
        process_1 = {}
        process_1['exits']      = exits
        process_1['error_mod']  = error_mod
        process_1['error_sup']  = error_sup
        process_1['error_conv'] = error_conv

        # data_1
        data_1 = {}
        data_1['data'] = data

        # entry_1
        entry_1 = {}
        entry_1['sample_1'] = sample_1
        entry_1['instrument_1'] = instrument_1
        entry_1['image_1'] = image_1
        entry_1['data_1'] = data_1

        # root
        self.root['cxi_version'] = 140
        self.root['entry_1'] = entry_1

    def write(self, fnam):
        """Put everything in root into a h5file
        
        Recurse through self.root then add the soft links at
        the end."""
        f = h5py.File(fnam, "w")
        
        def mywrite(d, f1):
            for k, v in d.iteritems():
                if isinstance(v, dict):
                    f2 = f1.create_group(k)
                    print "Creating group: {0} <-- {1} ".format(f1.name, k)
                    mywrite(v, f2)
                    
                elif isinstance(v, numbers.Number) or isinstance(v, str):
                    f1.create_dataset(k, data = v)
                    print "Adding data: {0} <-- {1}, {2}".format(f1.name, k, str(v))
                    
                elif isinstance(v, np.ndarray):
                    if len(v.shape) == 3 :
                        dset = f1.create_dataset(k, v.shape, dtype=v.dtype, chunks=(1, v.shape[1], v.shape[2]), compression='gzip')
                        dset[:] = v
                        print "Adding data: {0} <-- {1}, {2} {3}".format(f1.name, k, v.dtype, v.shape)
                        
                    elif len(v.shape) <= 2 :
                        dset = f1.create_dataset(k, v.shape, dtype=v.dtype, chunks=v.shape, compression='gzip')
                        dset[:] = v
                        print "Adding data: {0} <-- {1}, {2} {3}".format(f1.name, k, v.dtype, v.shape)
                        
                    else :
                        print "Warning max dimensions = 3, {1} {2}".format(k, v.shape)
        mywrite(self.root, f)
        
        # now add the soft links and attributes
        f['entry_1/image_1/data'].attrs['axes'] = ['translation:y:x']
        f['entry_1/image_1/translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')
        f['entry_1/image_1/instrument_1'] = h5py.SoftLink('/entry_1/instrument_1')
        
        f['entry_1/data_1/data'] = h5py.SoftLink('/entry_1/image_1/data')
        print ''
        print ''
        bg.print_h5(f)
        f.close()
        print "done!"

    def check(self):
        # recursively search through the data structure
        def myprint(d):
            for k, v in d.iteritems():
                if isinstance(v, dict):
                    myprint(v)
                elif v == None :
                    print "Warning {0} : {1}".format(k, v)
        myprint(self.root)


class CXI_file(object):
    """Ptychography data --> *.cxi file 

    - Class that mimics the cxi h5 dataset for Ptychographic data.
    - This is just a class that enforces the cxi file format on the input data.
    - SI units only.
    """

    def __init__(self):
        # Construct the data structure
        self.root = {}
        
        # sample_1
        geometry_1 = {}
        geometry_1['translation'] = None
        
        sample_1 = {}
        sample_1['name'] = None
        sample_1['geometry_1'] = geometry_1
        
        # instrument_1
        detector_1 = {}
        detector_1['distance'] = None
        detector_1['corner_position'] = None
        detector_1['x_pixel_size'] = None
        detector_1['y_pixel_size'] = None
        
        source_1 = {}
        source_1['energy'] = None
        source_1['probe'] = None
        source_1['probe_mask'] = None
        
        instrument_1 = {}
        instrument_1['detector_1'] = detector_1
        instrument_1['source_1'] = source_1
        
        # image_1
        image_1 = {}
        image_1['data'] = None
        image_1['translation'] = None
        image_1['intrument_1'] = None
        
        # data_1
        data_1 = {}
        data_1['data'] = None

        # entry_1
        entry_1 = {}
        entry_1['sample_1'] = sample_1
        entry_1['instrument_1'] = instrument_1
        entry_1['image_1'] = image_1
        entry_1['data_1'] = data_1

        # root
        self.root['cxi_version'] = 140
        self.root['entry_1'] = entry_1

    def write(self, fnam):
        """Put everything in root into a h5file
        
        Recurse through self.root then add the soft links at
        the end."""
        f = h5py.File(fnam, "w")
        
        def mywrite(d, f1):
            for k, v in d.iteritems():
                if isinstance(v, dict):
                    f2 = f1.create_group(k)
                    print "Creating group: {0} <-- {1} ".format(f1.name, k)
                    mywrite(v, f2)
                    
                elif isinstance(v, numbers.Number) or isinstance(v, str):
                    f1.create_dataset(k, data = v)
                    print "Adding data: {0} <-- {1}, {2}".format(f1.name, k, str(v))
                    
                elif isinstance(v, np.ndarray):
                    if len(v.shape) == 3 :
                        dset = f1.create_dataset(k, v.shape, dtype=v.dtype, chunks=(1, v.shape[1], v.shape[2]), compression='gzip')
                        dset[:] = v
                        print "Adding data: {0} <-- {1}, {2} {3}".format(f1.name, k, v.dtype, v.shape)
                        
                    elif len(v.shape) <= 2 :
                        dset = f1.create_dataset(k, v.shape, dtype=v.dtype, chunks=v.shape, compression='gzip')
                        dset[:] = v
                        print "Adding data: {0} <-- {1}, {2} {3}".format(f1.name, k, v.dtype, v.shape)
                        
                    else :
                        print "Warning max dimensions = 3, {1} {2}".format(k, v.shape)
        mywrite(self.root, f)
        
        # now add the soft links and attributes
        f['entry_1/image_1/data'].attrs['axes'] = ['translation:y:x']
        f['entry_1/image_1/translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')
        f['entry_1/image_1/instrument_1'] = h5py.SoftLink('/entry_1/instrument_1')
        
        f['entry_1/data_1/data'] = h5py.SoftLink('/entry_1/image_1/data')
        print ''
        print ''
        bg.print_h5(f)
        f.close()
        print "done!"

    def check(self):
        # recursively search through the data structure
        def myprint(d):
            for k, v in d.iteritems():
                if isinstance(v, dict):
                    myprint(v)
                elif v == None :
                    print "Warning {0} : {1}".format(k, v)
        myprint(self.root)


class Pcxi_view(object):
    """Use pyqtgraph to look at a Ptychography_cxi file"""
    def __init__(self):
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
    f['entry_1/sample_1'].create_dataset('Transmission function', data = data['sample'])
    f['entry_1/sample_1'].create_group('geometry_1')
    f['entry_1/sample_1/geometry_1'].create_dataset('translation', data = data['sample_translations_xyz'])

    f['entry_1'].create_group('instrument_1')
    f['entry_1/instrument_1'].create_group('detector_1')
    z     = data['z']
    shape = data['diffs'].shape
    f['entry_1/instrument_1/detector_1'].create_dataset('distance', data = z)
    f['entry_1/instrument_1/detector_1'].create_dataset('corner_position', data = np.array([ (shape[1]-3)/2., (shape[0]-3)/2., z]))
    f['entry_1/instrument_1/detector_1'].create_dataset('y_pixel_size', data = data['dr'][0])
    f['entry_1/instrument_1/detector_1'].create_dataset('x_pixel_size', data = data['dr'][0])
    f['entry_1/instrument_1'].create_group('source_1')
    f['entry_1/instrument_1/source_1'].create_dataset('probe_mask', data = np.ones_like(data['probe'], dtype=np.bool))
    f['entry_1/instrument_1/source_1'].create_dataset('energy', data = data['energy'])
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
    
    cxi_file = Pcxi(cxi_file)
    
    Pcxi.ERA(cxi_file, iters=10, update='sample')

