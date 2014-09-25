import os, sys, getopt, inspect
import numpy as np
import h5py
import scipy as sp
import STEM_probe 
from scipy.optimize import curve_fit
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import convolve

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from python_scripts import bagOfns as bg

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

import h5py

f = h5py.File(fileName, "w")
f.create_dataset("cxi_version",data=140)
entry_1 = f.create_group("entry_1")
entry_1.create_dataset("experiment_identifier", data = 'Peng CeO2 data')

sample_1 = entry_1.create_group("sample_1")
sample_1.create_dataset("name", data = 'CeO2')
sample_1.create_dataset("description", data = 'cerium dioxide nanoparticle. Cubeoctahedron with limited truncation along {100}.')
sample_1.create_dataset("thickness", data = 50.0e-10)


