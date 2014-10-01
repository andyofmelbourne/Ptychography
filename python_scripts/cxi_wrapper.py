import numpy as np
from Ptychography_2dsample_2dprobe_farfield import Ptychography

class Pcxi(object):
    """A wrapper class that applies the functionality of the Ptychography class to cxi files.
    """
    def __init__(self, cxi_fnam):
        """Initialise the Ptychography class from the cxi file
        
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
        # data structure in the form of dicts
        pass
    
    def write(self, fnam):
        pass

    def read(self, fnam):
        pass


class P_to_cxi(object):

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



class Pcxi_view(object):
    """Use pyqtgraph to look at a Ptychography_cxi file"""
    def __init__(self):
        pass


