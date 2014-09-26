import os, sys, getopt, inspect
import numpy as np
import h5py
import scipy as sp
import STEM_probe 
from scipy.optimize import curve_fit
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import convolve
import scipy.constants
import h5py
import numbers

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from python_scripts import bagOfns as bg

class CXI_file(object):
    """Class that mimics the cxi h5 dataset for Ptychographic data

    SI units only"""

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

def gauss(x, *p):
	a, mu, sigma = p
	return a*np.exp(-(x-mu)**2/(2.*sigma**2))

def fit_gaus(hist_0, return_fit = True):
	"""Fit a gaussian to the 1d numpy array "hist" and return the parameters

	gaus = a exp( - (x - mu)^2 / 2 sigma^2)
	where x is in units of pixels
	returns a, mu, sigma
	"""
	hist = hist_0.astype(np.float64)
	bins = np.arange(hist.shape[0]).astype(np.float64)
	
	# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
	p0 = [hist.max(), float(np.argmax(hist)), 100.0]
	
	coeff, var_matrix = curve_fit(gauss, bins, hist, p0=p0)
	
	if return_fit:
		# Get the fitted curve
		hist_fit = gauss(bins, *coeff)
		return hist_fit, coeff[0], coeff[1], coeff[2]
	else :
		return coeff[0], coeff[1], coeff[2]



# parameters
#E = 300kV
#24 mrad (~12nm-1)
#defocus = 91 nm (overfocus)


# I would like to process the data then put it into a h5file for viewing and such


# get diffs
diffs = np.empty((7*7, 1024, 1024), dtype=np.float32)

dirnam = '/home/amorgan/Physics/rawdata/CeO2_Ptych_data/dat_files/'
fnams = bg.get_fnams(start = '', dir_base = dirnam, end = '')
fnams = np.sort(fnams)

for i in range(len(fnams)):
    diffs[i] = bg.binary_in(fnams[i], dim = diffs.shape[1 :], dt = diffs.dtype, endianness='little')

# get coords (probe position coords)
fnam = '/home/amorgan/Physics/rawdata/CeO2_Ptych_data/c_list.txt'

d = (float, float)
x, y = np.loadtxt(fnam, dtype=d, unpack=True)
x *= diffs.shape[-1]
y *= diffs.shape[-2]

# Let's reorder these for sample translations (instead of probe translations)
x *= -1
y *= -1
sample_translations_xyz = np.array( zip(np.zeros_like(x), x, y) )

# stack the diffs for display
diffs_v = None
for i in range(6, -1, -1):
    diffs_h = []
    for j in range(7):
        diffs_h.append(bg.izero_pad(diffs[i * 7 + j], (512, 512))[::2,::2])
    diffs_h = np.hstack(tuple(diffs_h))
    if diffs_v != None :
        diffs_v = np.vstack((diffs_v, diffs_h))
    else :
        diffs_v = diffs_h
diffs_montage = diffs_v

# reorder diffraction data accordingly
diffs_old = diffs.copy()
diffs = []
for i in range(6, -1, -1):
    for j in range(7):
        diffs.append(diffs_old[i * 7 + j])
diffs = np.array(diffs)

# fit a gaussian to the background
bins = range(int(diffs.min() - 1), int(diffs.max())+1)
hist, bins = np.histogram(diffs, bins)
hist_fit, A, mu, sigma = fit_gaus(hist, return_fit = True)

# threshold the data to 4 sigma
min_count = mu + bins.min() + 4 * sigma 
diffs = diffs - min_count
diffs *= diffs > 0

# find bad pix
diffs_sum = np.sum(diffs, axis=0)
diffs_mf  = median_filter(diffs_sum, footprint = np.ones((6, 6)))
#diffs_mf  = convolve(diffs_sum, np.ones((16, 16))/16.**2)

mask = np.zeros_like(diffs_sum)
mask = (diffs_mf == 0) * (diffs_sum > 10) 
mask = ~mask

diffs = diffs * mask

# make the probe 
app_mask  = bg.blurthresh_mask(diffs[0])
aperture  = app_mask * np.sqrt(diffs[0])

probe = STEM_probe.makePupilPhase(aperture)


# put it all into a h5 file
fileName = "CeO2_Physical_Review_2014.cxi"

prob_cxi = CXI_file()
prob_cxi.root['entry_1']['sample_1']['name'] = 'CeO2'
prob_cxi.root['entry_1']['sample_1']['description'] = 'cerium dioxide nanoparticle. Cubeoctahedron with limited truncation along {100}.'
prob_cxi.root['entry_1']['sample_1']['thickness'] = 50.0e-10
prob_cxi.root['entry_1']['sample_1']['geometry_1']['translation'] = sample_translations_xyz

du = 28.0e-6 # pixel size
z = du / (probe.dq * probe.lamb)
prob_cxi.root['entry_1']['instrument_1']['detector_1']['distance'] = z
prob_cxi.root['entry_1']['instrument_1']['detector_1']['corner_position'] = np.array([ (diffs.shape[1]-3)/2., (diffs.shape[0]-3)/2., z])
prob_cxi.root['entry_1']['instrument_1']['detector_1']['x_pixel_size'] = du
prob_cxi.root['entry_1']['instrument_1']['detector_1']['y_pixel_size'] = du
prob_cxi.root['entry_1']['instrument_1']['source_1']['energy'] = probe.energy * scipy.constants.e
prob_cxi.root['entry_1']['instrument_1']['source_1']['probe'] = probe.probeR
prob_cxi.root['entry_1']['instrument_1']['source_1']['probe_mask'] = np.ones_like(probe.probeR, dtype=np.bool)

prob_cxi.root['entry_1']['image_1']['data'] = diffs

CXI_file.check(prob_cxi)
