import numpy as np
from scipy import ndimage
from scipy import fftpack
import scipy.misc as sm
import scipy as sp
from random import *
import random
import time
import threading
import sys, os, getopt
from ctypes import *

####################################################################################
####################################################################################
# file input and output routines. As well as h5 support
# Input/Output Routines
def binary_in(fnam, dim = (0,0), dt = np.dtype(np.float64), endianness='big', dimFnam = False):
    """Read a 2-d array from a binary file. 
    
    if dimFnam == True then the array dimensions are deduced from the file name but only if they have the form
    fnam_10x30x20.raw
    where raw may be replaced by any three letter string."""
    if dimFnam :
        fnam_base  = os.path.basename(fnam)
        fnam_dir   = os.path.abspath(os.path.dirname(fnam))
        onlyfiles  = [ f for f in os.listdir(fnam_dir) if os.path.isfile(os.path.join(fnam_dir,f)) ]
        fnam_match = [ f for f in onlyfiles if f[:len(fnam_base)] == fnam_base ]
        try : 
            fnam_match[0]
        except :
            raise NameError('I can\'t seem to find this file you speak of...\t'+fnam)
        dim        = fnam_match[0][fnam_match[0].rfind('_') + 1 :-4]
        #dim        = fnam_match[0][len(fnam_base) + 1 :-4]
        dim_list   = []
        b = ''
        for a in dim:
            if a != '_' and a != 'x' :
                b += a
            else :
                dim_list.append(int(b))
                b = ''
        dim_list.append(int(b))
        dim     = tuple(dim_list)
        fnam_in = fnam_match[0]
        fnam_in = os.path.join(fnam_dir, fnam_in)
        arrayout = np.fromfile(os.path.abspath(fnam_in),dtype=dt).reshape( dim )
    else :
        arrayout = np.fromfile(os.path.abspath(fnam),dtype=dt).reshape( dim )
    
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    return arrayout

def binary_out(array, fnam, dt=np.dtype(np.float64), endianness='big', appendDim=False):
    """Write a 2-d array to a binary file.
    
    If appendDim == True then the dimensions of the output array are appended to the file name.
    eg. fnam --> fnam_10x20x30.raw
    This is done so that binary_in with dimFnam == True will work."""
    if appendDim == True :
        fnam_out = fnam + '_'
        for i in array.shape[:-1] :
            fnam_out += str(i) + 'x' 
        fnam_out += str(array.shape[-1]) + '.raw'
    else :
        fnam_out = fnam
    arrayout = np.array(array, dtype=dt)
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    arrayout.tofile(os.path.abspath(fnam_out))

def get_fnams(start = '', dir_base = './', end = ''):
    """Return the relative path and file names off all files in 'dir_base' that start with 'base' and end with 'end'."""
    fnams           = os.listdir(dir_base)
    fnams_out       = []
    for i, fnam in enumerate(fnams):
        if fnam[:len(start)] == start :
            if fnam[-len(end):] == end or len(end) == 0 :
                temp = os.path.join( dir_base, fnam)
                if os.path.isfile( temp ) :
                    fnams_out.append(temp)
    return fnams_out
    
def fnamBase_match(fnam):
    fnam_base  = os.path.basename(fnam)
    fnam_dir   = os.path.abspath(os.path.dirname(fnam))
    onlyfiles  = [ f for f in os.listdir(fnam_dir) if os.path.isfile(os.path.join(fnam_dir,f)) ]
    fnam_match = [ f for f in onlyfiles if f[:len(fnam_base)] == fnam_base ]
    try : 
        fnam_match[0]
    except :
        return False
    return True
####################################################################################
####################################################################################
def quadshift(a):
    """Move N/2 - 1 to 0 for the last two dimensions. If the array is 1d then this is only done over the first."""
    if len(a.shape) == 1 :
        b = np.roll(a, -(a.shape[-1]/2-1), -1)
    else :
        b = np.roll(a, -(a.shape[-2]/2-1), -2)
        b = np.roll(b, -(b.shape[-1]/2-1), -1)
    return b

def iquadshift(a):
    """Move 0 to N/2 - 1 for the last two dimensions. If the array is 1d then this is only done over the first."""
    if len(a.shape) == 1 :
        b = np.roll(a, +(a.shape[-1]/2-1), -1)
    else :
        b = np.roll(a, +(a.shape[-2]/2-1), -2)
        b = np.roll(b, +(b.shape[-1]/2-1), -1)
    return b

def fft2(a, origin='centre'):
    """Norm preserving fft on a over the last two dimensions (2d ffts). 
    
    If len(a) == 1 then a 1d transform is done.
    If len(a) == 2 then a 2d transform is done.
    If len(a) >  2 then many 2d transforms are done.
    
    If origin is 'zero' then the the zero pixel is at 0,0.
    If origin is 'centre' then the the zero pixel is at a.shape[0]/2 -1, a.shape[0]/2 -1.
    
    this is not an in place operation."""
    if origin == 'centre' :
        b = quadshift(a)
    else :
        b = a.copy()
    if len(b.shape) == 1 :
        b = np.fft.fftpack.fft(b)     
    elif len(b.shape) == 2 :
        b = np.fft.fftpack.fft2(b)     
    elif len(b.shape) > 2 :
        b = fftn(b)
        if origin == 'centre' :
            b = iquadshift(b)
        return b
    if origin == 'centre' :
        b = iquadshift(b)
    return np.divide(b, np.sqrt(b.size))

def ifft2(a, origin='centre'):
    """Norm preserving inverse fft on a over the last two dimensions (2d ffts). 
    
    If len(a) == 1 then a 1d transform is done.
    If len(a) == 2 then a 2d transform is done.
    If len(a) >  2 then many 2d transforms are done.
    
    If origin is 'zero' then the the zero pixel is at 0,0.
    If origin is 'centre' then the the zero pixel is at a.shape[0]/2 -1, a.shape[0]/2 -1.
    
    this is not an in place operation."""
    if origin == 'centre' :
        b = quadshift(a)
    else :
        b = a.copy()
    if len(b.shape) == 1 :
        b = np.fft.fftpack.ifft(b)     
    elif len(b.shape) == 2 :
        b = np.fft.fftpack.ifft2(b)     
    elif len(b.shape) > 2 :
        b = ifftn(b)
        if origin == 'centre' :
            b = iquadshift(b)
        return b
    if origin == 'centre' :
        b = iquadshift(b)
    return np.multiply(b, np.sqrt(b.size))

def fftn(a):
    """Norm preserving fft on the zero pixel 0,0 this is not in place."""
    b = np.fft.fftpack.fftn(a, axes=(len(a.shape)-2,len(a.shape)-1))     
    return np.divide(b, np.sqrt(a.shape[-1] * a.shape[-2]))

def ifftn(a):
    """Norm preserving fft on the zero pixel 0,0 this is not in place."""
    b = np.fft.fftpack.ifftn(a, axes=(len(a.shape)-2,len(a.shape)-1))     
    return np.multiply(b, np.sqrt(a.shape[-1] * a.shape[-2]))

def fftn_1d(a):
    """Norm preserving fft on the zero pixel 0,0 this is not in place."""
    b = np.fft.fftpack.fftn(a, axes=(len(a.shape)-1, ))     
    return np.divide(b, np.sqrt(a.shape[-1]))

def ifftn_1d(a):
    """Norm preserving fft on the zero pixel 0,0 this is not in place."""
    b = np.fft.fftpack.ifftn(a, axes=(len(a.shape)-1, ))     
    return np.multiply(b, np.sqrt(a.shape[-1]))

####################################################################################
####################################################################################
# I want thresholding masking and such. Maybe look at numpy's masked array operations
def threshold(arrayin, thresh = 1.0):
    """Threshold any values in arrayin above thresh to "thresh", apply this to the amplitude only for complex arrays."""
    if arrayin.dtype == 'complex' :
        arrayout = np.abs(arrayin)
    else :
        arrayout = arrayin
    mask     = np.array(1.0 * (arrayout > thresh),dtype=np.bool)  
    arrayout = (~mask) * arrayout + mask * thresh
    if arrayin.dtype == 'complex' :
        arrayout = arrayout * np.exp(1J*np.angle(arrayin))
    return arrayout

def gauss(arrayin, a, ryc=0.0, rxc=0.0): 
    """Return a real gaussian as an numpy array e^{-a x^2}."""
    ny = arrayin.shape[0]
    nx = arrayin.shape[1]
    # ryc and rxc are the coordinates of the center of the gaussian
    # in fractional unints. so ryc = 1 rxc = 1 puts the centre at the 
    # bottom right and -1 -1 puts the centre at the top left
    shifty = int(ryc * ny/2 - 1)
    shiftx = int(rxc * nx/2 - 1)
    arrayout = np.zeros((ny,nx))
    for i in range(0,ny):
        for j in range(0,nx):
            x = np.exp(-a*((i-ny/2 + 1)**2 + (j-nx/2 + 1)**2))
            arrayout[i][j] = x

    if ryc != 0.0 :
        arrayout = np.roll(arrayout,shifty,0)
    if rxc != 0.0 :
        arrayout = np.roll(arrayout,shiftx,1)

    return arrayout

def gauss_2d(N, sigma = 0.25):
    """Return a gaussian with a one on e value (sigma) as a fraction of the array.
    
    Normalised so that sum gauss_n approx 1
    N.B. this is approximate for speed. """
    x, y     = make_xy(N)
    sigma_pixel = sigma * np.float(N)
    arrayout = np.exp(-(x**2 + y**2) / sigma_pixel**2) / (np.pi * sigma_pixel**2)
    return arrayout

def gauss_1d(arrayin,a,ryc=0.0,rxc=0.0): 
    """Return a real gaussian as an numpy array e^{-a x^2}."""
    ny = arrayin.shape[0]
    # ryc and rxc are the coordinates of the center of the gaussian
    # in fractional unints. so ryc = 1 rxc = 1 puts the centre at the 
    # bottom right and -1 -1 puts the centre at the top left
    shifty   = int(ryc * ny/2 - 1)
    arrayout = np.zeros((ny))
    for i in range(0,ny):
        x = np.exp(-a*((i-ny/2 + 1)**2))
        arrayout[i] = x

    if ryc != 0.0 :
        arrayout = np.roll(arrayout,shifty,0)
    return arrayout

def circle_new(shape = (1024, 1024), radius=0.25, Nrad = None, origin=[0,0]):
    """Make a circle of optional radius as a fraction of the array size.
    
    If ny > nx (or vise versa) use the smaller radius."""
    if Nrad == None :
        pass
    else :
        radius = max([shape[0], shape[1]]) 
        radius = np.float(Nrad) / np.float(radius) 
    # 
    x, y = make_xy(shape, origin = origin)
    r    = np.sqrt(x**2 + y**2)
    if shape[1] > shape[0]:
        rmax = radius * shape[0] / 2
    else :
        rmax = radius * shape[1] / 2
    arrayout = (r <= rmax)
    return np.array(arrayout, dtype=np.float64)

def circle(ny, nx, radius=0.25, Nrad = None):
    """Make a circle of optional radius as a fraction of the array size"""
    if Nrad == None :
        nrad     = (ny * radius)**2
        arrayout = np.zeros((ny,nx))
        for i in range(0,ny):
            for j in range(0,nx):
                r = (i - ny/2 + 1)**2 + (j - nx/2 + 1)**2
                if r < nrad:
                    arrayout[i][j] = 1.0
    else :
        nrad     = Nrad**2
        arrayout = np.zeros((ny,nx))
        for i in range(0,ny):
            for j in range(0,nx):
                r = (i - ny/2 + 1)**2 + (j - nx/2 + 1)**2
                if r < nrad:
                    arrayout[i][j] = 1.0
    return arrayout

def waveno(E):
    """Calculates the wave-number of an electron from its energy.

    Units are in eV and Angstroms."""
    C1 = 9.7846113e-07
    C2 = 12.263868e0 
    k = np.sqrt(E + C1 * E**2)/C2
    return k

def energyK(k):
    """Calculates the energy of an electron from its wave-number.

    Units are in eV and Angstroms."""
    C1 = 9.7846113e-07
    C2 = 12.263868e0 
    E = (-1.0 + np.sqrt(1.0 + 4.0 * C1 * C2**2 * k**2))/(2.0 * C1)
    return E

def mapping(array, map):
    """Tile the 2d array into a real 1d array via map."""
    n = map.shape[0] 
    if array.dtype == np.complex :
        arrayout = np.zeros((2*n),dtype=array.real.dtype)
        for ii in range(n):
            i = map[ii,0]
            j = map[ii,1]
            arrayout[ii]   = array[i,j].real
            arrayout[ii+n] = array[i,j].imag
    else :
        arrayout = np.zeros((n),dtype=array.dtype)
        for ii in range(n):
            i = map[ii,0]
            j = map[ii,1]
            arrayout[ii]   = array[i,j]
    return arrayout

def unmapping(array, map, N):
    """Untile the 1d real array into a 2d array via map."""
    n = map.shape[0] 
    if (array.shape[0] / 2) == N**2 :
        arrayout = np.zeros((N,N),dtype=np.complex128)
        for ii in range(n):
            i = map[ii,0]
            j = map[ii,1]
            arrayout[i,j]   = array[ii] + 1.0J * array[ii+n]
    elif array.shape[0] == N**2 :
        arrayout = np.zeros((N,N),dtype=array.dtype)
        for ii in range(n):
            i = map[ii,0]
            j = map[ii,1]
            arrayout[i,j]   = array[ii] 
    return arrayout

def autoc(array):
    """Return the auto-correlation function of array."""
    return ifft2(np.square(np.abs(fft2(array))))

def correlate(array1,array2):
    """Return the cross-correlation function of array1 and array2."""
    arrayout = np.conj(fft2(array1)) * fft2(array2)
    return ifft2(arrayout)

def dblefy(fnam,N,dt=np.float32,endianness='big',fnam2=None):
    """Load in fnam then convert it to dtype('float64') and overwrite the original file."""
    array  = binary_in(fnam,N,N,dt,endianness)
    array2 = np.array(array,dtype=np.float64)
    if fnam2 == None : fnam2 = fnam
    binary_out(array2,fnam2,dt=np.float64,endianness='big')

def addNoise_amp(array,counts):
    """Add poisson noise to an array.

    The mean number at each pixel is the value at that
    pixel times the number of photons or 'trials' 
    accross the whole array.
    """
    if array.dtype == 'complex' :
        arrayout = addNoise(np.real(array),counts) + 1.0J * addNoise(np.imag(array),counts)
    else :
        if np.float64(counts) == 0.0e0 :
            arrayout = np.copy(array)
        elif np.float64(counts) < 0.0e0 :
            print 'bg.addNoise : warning counts < 0'
        else :
            arrayout = np.zeros(array.shape)
            arrayout = np.square(normalise(array))
            arrayout = np.random.poisson(arrayout*np.float64(counts))/np.float64(counts)
            arrayout = np.sqrt(arrayout)
            tot      = np.sum(np.abs(array)**2)
            arrayout = normalise(arrayout,tot)
    return arrayout

def addNoise(array,counts):
    """Add poisson noise to an array.

    The mean number at each pixel is the value at that
    pixel times the number of photons or 'trials' 
    accross the whole array.
    """
    if array.dtype == 'complex' :
        arrayout = addNoise(np.real(array),counts) + 1.0J * addNoise(np.imag(array),counts)
    else :
        if np.float64(counts) == 0.0e0 :
            arrayout = np.zeros(array.shape, dtype=arrayout.dtype)
        elif np.float64(counts) < 0.0e0 :
            print 'bg.addNoise : warning counts < 0'
        elif np.float64(counts) > 1.0e9 :
            arrayout = np.zeros(array.shape)
            arrayout = normaliseInt(array)
            arrayout = np.random.normal(arrayout*np.float64(counts),np.sqrt(arrayout*np.float64(counts)))/np.float64(counts)
            tot      = np.sum(array)
            arrayout = normaliseInt(arrayout,tot)
        else :
            arrayout = np.zeros(array.shape)
            arrayout = normaliseInt(array)
            arrayout = np.random.poisson(arrayout*np.float64(counts))/np.float64(counts)
            tot      = np.sum(array)
            arrayout = normaliseInt(arrayout,tot)
    return arrayout

def normalise(array,tot=1.0):
    """normalise the array to tot.

    normalises such that sum |arrayout|^2 = tot.
    """
    tot1 = np.sum(np.abs(array)**2)
    if tot1 == 0.0 :
        print 'bg.normalise : warning sum array = 0'
        arrayout = np.copy(array)
    else :
        arrayout = array * np.sqrt(tot / tot1)
    return arrayout

def normaliseInt(array,tot=1.0):
    """normalise the array to tot.

    normalises such that sum arrayout = tot.
    """
    tot1 = np.sum(array)
    arrayout = array * tot / tot1
    return arrayout

def l2norm(array1,array2):
    """Calculate sqrt ( sum |array1 - array2|^2 / sum|array1|^2 )."""
    tot = np.sum(np.abs(array1)**2)
    return np.sqrt(np.sum(np.abs(array1-array2)**2)/tot)

def plotPower(array):
    """Take the autocorrelation of array and plot the radialy averaged value.
    
    map each pixel to a discrete r value
    count the number of pixels at each r
    fr = average array(r) value
    """
    ny, nx    = array.shape
    nyc = ny/2 - 1
    nxc = nx/2 - 1
    array2, r, theta = reproject_image_into_polar(array, [nyc,nxc])
    # Let's average over theta
    array3 = np.zeros(ny)
    for i in range(ny):
        array3[i] = np.sum(array2[i,:])/np.float(nx)
    
    return array3

def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None :
        origin_x, origin_y = nx // 2, ny // 2
    else :
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def make_xy(N, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image (N/2 - 1, N/2 - 1). 
    Specify origin=(0,0) to set the origin to the lower left corner 
    of the image."""
    if type(N) == int :
        ny, nx = N, N
    elif len(N) == 2 :
        ny, nx = N 
    elif len(N) == 1 and (type(N) == tuple or type(N) == list):
        ny = N[0] 
        nx = 0
    if ny != 0 and nx != 0 :
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        if (origin is None) or (origin == 'centre') :
            origin_x, origin_y = nx // 2 - 1, ny // 2 - 1
            x -= origin_x
            y -= origin_y
        elif origin[0] == 0 and (origin[1] == 0):
            x = ((x + nx // 2 - 1) % nx) - (nx // 2 - 1)
            y = ((y + ny // 2 - 1) % ny) - (ny // 2 - 1)
        return x, y
    elif ny != 0 and nx == 0 :
        y = np.arange(ny)
        if (origin is None) or (origin == 'centre') :
            origin_y = ny // 2 - 1
            y -= origin_y
        elif origin[0] == 0 :
            y = ((y + ny // 2 - 1) % ny) - (ny // 2 - 1)
        return y

def cart2polar(x, y):
    r     = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def reproject_image_into_polar(data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    r, theta = cart2polar(x, y)

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i                = np.linspace(r.min(), r.max(), nx)
    theta_i            = np.linspace(theta.min(), theta.max(), ny)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)
    xi    += origin[0] # We need to shift the origin back to 
    yi    += origin[1] # back to the lower-left corner...
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)

    # Reproject each band individually and the restack
    # (uses less memory than reprojection the 3-dimensional array in one step)
    bands = []
    zi = sp.ndimage.map_coordinates(data, coords, order=1)
    return zi.reshape((nx, ny)), r_i, theta_i


def centre(arrayin):
    """Shift an array so that its centre of mass is on the N/2 - 1."""
    ny = arrayin.shape[0]
    nx = arrayin.shape[1]
    cy = 0.0
    cx = 0.0
    for i in range(ny):
        for j in range(nx):
            cy += np.float64(arrayin[i,j]) * np.float64(i - ny/2 + 1)
            cx += np.float64(arrayin[i,j]) * np.float64(j - nx/2 + 1)
    cx = cx / np.sum(arrayin)
    cy = cy / np.sum(arrayin)
    arrayout = np.roll(arrayin ,-int(cy),0)
    arrayout = np.roll(arrayout,-int(cx),1)
    return [arrayout,cy,cx]

def thresh_mask(arrayin, thresh=0.1e0):
    """Threshold arrayin to thresh of its maximum and return a true/flase mask. If arrayin is complex the amplitude is used as the threshold array."""
    if arrayin.dtype == 'complex' :
        arrayout = np.abs(arrayin)
    else :
        arrayout = arrayin
    thresh2  = np.max(arrayout)*thresh
    arrayout = np.array(1.0 * (arrayout > thresh2),dtype=np.bool)  
    return arrayout

def blur(arrayin, blur=8):
    """gaussian blur arrayin."""
    arrayout = np.array(arrayin,dtype=np.float64)
    arrayout = ndimage.gaussian_filter(arrayout,blur)
    arrayout = np.array(arrayout,dtype=arrayin.dtype)  
    return arrayout

def blurthresh(arrayin,thresh=0.1e0,blur=8):
    """gaussian blur then threshold to expand the region.
    
    Output a True/False mask."""
    arrayout = np.array(arrayin,dtype=np.float64)
    arrayout = ndimage.gaussian_filter(arrayout,blur)
    thresh2  = np.max(np.abs(arrayout))*thresh
    arrayout = np.array(1.0 * (np.abs(arrayout) > thresh2),dtype=arrayin.dtype)  
    return arrayout

def blurthresh_mask(arrayin,thresh=0.1e0,blur=8):
    """gaussian blur then threshold to expand the region.
    
    Output a True/False mask."""
    arrayout = np.array(arrayin,dtype=np.float64)
    arrayout = ndimage.gaussian_filter(arrayout,blur)
    thresh2  = np.max(np.abs(arrayout))*thresh
    arrayout = np.array(1.0 * (np.abs(arrayout) > thresh2),dtype=np.bool)  
    return arrayout

def threshExpand(arrayin,thresh=0.1e0,blur=8):
    """Threshold the array then gaussian blur then rethreshold to expand the region.
    
    Output a True/False mask."""
    arrayout = np.array(arrayin,dtype=np.float64)
    #arrayout = padd(arrayout,2*arrayin.shape[0])
    arrayout = ndimage.gaussian_filter(arrayout,blur)
    thresh2  = np.max(np.abs(arrayout))*thresh
    arrayout = 1.0 * (np.abs(arrayout) > thresh2)
    
    arrayout = ndimage.gaussian_filter(arrayout,2*blur)
    thresh2  = np.max(np.abs(arrayout))*thresh
    arrayout = np.array(1.0 * (np.abs(arrayout) > thresh2),dtype=np.bool)  
    #arrayout = unpadd(arrayout,arrayin.shape[0])
    return arrayout


def zero_pad(arrayin, shape = (1024, 1024), fillvalue = 0):
    """Padd arrayin with zeros until it is an ny*nx array, keeping the zero pixel at N/2-1
    
    only works when arrayin and arrayout have even domains."""
    if len(arrayin.shape) > 2 :
        shape_out     = list(arrayin.shape)
        shape_out[-1] = shape[1]
        shape_out[-2] = shape[0]
        arrayout = np.zeros(tuple(shape_out), dtype=arrayin.dtype)
        for i in range(len(arrayin)):
            arrayout[i] = zero_pad(arrayin[i], shape)
        return arrayout
    nyo = arrayin.shape[0]
    nxo = arrayin.shape[1]
    arrayout = np.zeros(shape, dtype = arrayin.dtype)
    arrayout.fill(fillvalue)
    arrayout[(shape[0]-nyo)//2 : (shape[0]-nyo)//2 + nyo,(shape[1]-nxo)//2 : (shape[1]-nxo)//2 + nxo] = arrayin
    return arrayout

def izero_pad(arrayin, shape = (1024, 1024)):
    """Strip arrayin until it is an (shape[0], shape[1]) array, keeping the zero pixel at N/2-1
    
    only works when arrayin and arrayout have even domains."""
    if len(arrayin.shape) > 2 :
        shape_out     = list(arrayin.shape)
        shape_out[-1] = shape[1]
        shape_out[-2] = shape[0]
        arrayout = np.zeros(tuple(shape_out), dtype=arrayin.dtype)
        for i in range(len(arrayin)):
            arrayout[i] = izero_pad(arrayin[i], shape)
        return arrayout
    shape0   = arrayin.shape
    arrayout = np.zeros(shape, dtype = arrayin.dtype)
    if len(arrayin.shape) == 2 :
        arrayout = arrayin[(shape0[0]-shape[0])//2 : (shape0[0]-shape[0])//2 + shape[0],(shape0[1]-shape[1])//2 : (shape0[1]-shape[1])//2 + shape[1]]
    elif len(arrayin.shape) == 1 :
        arrayout = arrayin[(shape0[0]-shape[0])//2 : (shape0[0]-shape[0])//2 + shape[0]]
    return arrayout

def interpolate_bigger(arrayin,ny,nx=None):
    """Fourier interpolate arraiyin onto an ny*nx grid (needs to be even-->even!)."""
    if nx == None :
        nx = ny
    arrayout = np.array(arrayin,dtype=np.complex128)
    arrayout = fft2(arrayout)
    arrayout = padd(arrayout,ny,nx)
    arrayout = ifft2(arrayout)
    return np.array(arrayout,dtype=arrayin.dtype)

def scale(arrayin,Amin,Amax,mask=None):
    """Scale arrayin so that its min max is Amin Amax."""
    if (mask==None) and (arrayin.max() - arrayin.min())!=0.0 :
        Bmax = arrayin.max()
        Bmin = arrayin.min()
    elif (arrayin.max() - arrayin.min())!=0.0 :
        ny = arrayin.shape[0]
        nx = arrayin.shape[1]
        Bmax = arrayin.min()
        Bmin = arrayin.max()
        for i in range(ny):
            for j in range(ny):
                if mask[i,j] > 0.5e0 :
                    if arrayin[i,j] < Bmin :
                        Bmin = arrayin[i,j]
                    if arrayin[i,j] > Bmax :
                        Bmax = arrayin[i,j]
    else :
        print "andrew.bagOfns.scale : warning (arrayin.max() - arrayin.min())=0.0 "
        return np.copy(arrayin)

    arrayout = (arrayin - Bmin)*(Amax - Amin) / (Bmax - Bmin) + Amin
    return arrayout

def crop_to_nonzero(arrayin, mask=None):
    """Crop arrayin to the smallest rectangle that contains all of the non-zero elements and return the result. If mask is given use that to determine non-zero elements.
    
    If arrayin is a list of arrays then all arrays are cropped according to the first in the list."""

    if type(arrayin) == np.ndarray :
        array = arrayin
    elif type(arrayin) == list :
        array = arrayin[0]

    if mask==None :
        mask = array
    #most left point 
    for i in range(mask.shape[1]):
        tot = np.sum(np.abs(mask[:,i]))
        if tot > 0.0 :
            break
    left = i
    #most right point 
    for i in range(mask.shape[1]-1,-1,-1):
        tot = np.sum(np.abs(mask[:,i]))
        if tot > 0.0 :
            break
    right = i
    #most up point 
    for i in range(mask.shape[0]):
        tot = np.sum(np.abs(mask[i,:]))
        if tot > 0.0 :
            break
    top = i
    #most down point
    for i in range(mask.shape[0]-1,-1,-1):
        tot = np.sum(np.abs(mask[i,:]))
        if tot > 0.0 :
            break
    bottom = i
    if type(arrayin) == np.ndarray :
        arrayout = array[top:bottom+1,left:right+1]
    elif type(arrayin) == list :
        arrayout = []
        for i in arrayin :
            arrayout.append(i[top:bottom+1,left:right+1])
    return arrayout
    
def roll(arrayin, shift = (0, 0), silent = True):
    """np.roll arrayin by dy in dim -2 and dx in dim -1. If arrayin is 1d then just do that.
    
    If the shift coordinates are of type float then the Fourier shift theorem is used."""
    arrayout = arrayin.copy()
    # if shift is integer valued then use np.roll
    if (type(shift[0]) == int) or (type(shift[0]) == np.int) or (type(shift[0]) == np.int32) or (type(shift[0]) == np.int64):
        if shift[-1] != 0 :
            if silent == False :
                print 'arrayout = np.roll(arrayout, shift[-1], -1)'
            arrayout = np.roll(arrayout, shift[-1], -1)
        # if shift is 1d then don't roll the other dim (if it even exists)
        if len(arrayout.shape) >= 2 :
            if shift[-2] != 0 :
                if silent == False :
                    print 'arrayout = np.roll(arrayout, shift[-2], -2)'
                arrayout = np.roll(arrayout, shift[-2], -2)
    # if shift is float valued then use the Fourier shift theorem
    elif (type(shift[0]) == float) or (type(shift[0]) == np.float32) or (type(shift[0]) == np.float64):
        # if shift is 1d
        if len(shift) == 1 :
            if silent == False :
                print 'arrayout = fftn_1d(arrayout)'
                print 'arrayout = arrayout * phase_ramp(arrayout.shape, shift, origin = (0, 0))'
                print 'arrayout = ifftn_1d(arrayout)'
            arrayout = fftn_1d(arrayout)
            arrayout = arrayout * phase_ramp(arrayout.shape, shift, origin = (0, 0))
            arrayout = ifftn_1d(arrayout)
        elif len(shift) == 2 :
            if silent == False :
                print 'arrayout = fftn(arrayout)'
                print 'arrayout = arrayout * phase_ramp(arrayout.shape, shift, origin = (0, 0))'
                print 'arrayout = ifftn(arrayout)'
            arrayout = fftn(arrayout)
            arrayout = arrayout * phase_ramp(arrayout.shape, shift, origin = (0, 0))
            arrayout = ifftn(arrayout)
    return arrayout

def roll_to(arrayin, y = 0, x = 0, centre = 'middle'):
    """np.roll arrayin such that the centre of the array is moved to y and x.
    
    Where the centre is N/2 - 1"""
    ny, nx = np.shape(arrayin)
    if centre == 'middle' :
        ny_centre = ny/2 - 1
        nx_centre = nx/2 - 1
    elif centre == 'zero' :
        ny_centre = 0
        nx_centre = 0
    
    if (y - int(y)) == 0 and (x - int(x)) == 0 :
        arrayout = roll(arrayin, dy = int(y) - ny_centre, dx = int(x) - nx_centre)
    elif (y - int(y)) > 0 or (x - int(x)) > 0 :
        arrayout  = fft2(arrayin)
        arrayout *= phase_ramp(arrayout.shape, [y - ny_centre, x - nx_centre])
        arrayout  = ifft2(arrayout)
    else :
        raise NameError('shift coordinates are niether int nor float (this is a bad thing).')
    return np.array(arrayout, dtype = arrayin.dtype)

def roll_nowrap(arrayin,dy = 0,dx = 0):
    """np.roll arrayin by dy in dim 0 and dx in dim 0."""
    if (dy != 0) or (dx != 0):
        arrayout = np.roll(arrayin,dy,0)
        arrayout = np.roll(arrayout,dx,1)
        if dy > 0 :
            arrayout[:dy,:] = 0.0
        elif dy < 0 :
            arrayout[dy:,:] = 0.0
        if dx > 0 :
            arrayout[:,:dx] = 0.0
        elif dx < 0 :
            arrayout[:,dx:] = 0.0
    else :
        arrayout = arrayin
    return arrayout

def interpolate(arrayin,shape=(256, 256)):
    """Interpolate to the grid."""
    if arrayin.dtype == 'complex' :
        Ln = interpolate(np.real(arrayin),shape) + 1.0J * interpolate(np.imag(arrayin),shape)
        #Ln = interpolate(np.abs(arrayin),new_res) * np.exp(1.0J * interpolate(np.angle(arrayin),new_res))
    else :
        coeffs    = ndimage.spline_filter(arrayin)
        rows,cols = arrayin.shape
        coords    = np.mgrid[0:rows-1:1j*shape[0],0:cols-1:1j*shape[1]]
        Ln        = sp.ndimage.map_coordinates(coeffs, coords, prefilter=False)
    return Ln

def bindown_tile(arrayin, new_res):
    """interpolate down arrayin to new_res then tile it. Keeps the zero pixel at N/2 -1."""
    N        = arrayin.shape[0]
    array    = interpolate(arrayin, new_res)
    array    = fft2(array)
    # find the smallest integer n such that n x new_res > N
    n = 0
    while n*new_res < N :
        n += 1
    M        = n*new_res
    array2   = np.zeros((M,M),dtype=np.complex128)
    
    for i in range(new_res):
        for j in range(new_res):
            ii = (i+1)*n - 1
            jj = (j+1)*n - 1
            array2[ii,jj] = array[i,j]
    
    array2   = ifft2(array2)
    arrayout = unpadd(array2,N) 
    arrayout = np.array(arrayout, dtype = arrayin.dtype)
    return arrayout 

def tile(arrayin, N, M = None):
    """Tile arrayin[Ny, Nx] by N times in a square arrayout[N x Ny, N x Nx]"""
    if M == None :
        M = N
    Ny, Nx = arrayin.shape
    arrayout = np.zeros((Ny * N, Nx * M), dtype = arrayin.dtype) 
    for i in range(N):
        for j in range(M):
            arrayout[i * Ny : (i+1) * Nx, j * Ny : (j+1) * Nx] = np.copy(arrayin)
    return arrayout

def orientate(arrayin,orientation):
    """Return a reorientated array.
    
    run through the 8 possibilities and output the corresponding exit wave
    f(x,y) :
    x,y x,-y -x,y -x,-y
    y,x y,-x -y,x -y,-x
    1   2    3    4 
    5   6    7    8 
    but we want to keep the zero pixel at N/2 - 1 
    so we will double the missing bit.
    so       0 1 2 3 
    becomes  2 1 0 0 
    NOT      3 2 1 0
    """
    ny = arrayin.shape[0]
    nx = arrayin.shape[1]
    
    if orientation == 1 :
        # x,y
        y = range(ny)
        x = range(nx)
        x, y = np.meshgrid(x,y)
    elif orientation == 2 :
        # x,-y
        y = range(ny-2,-1,-1)
        y.append(0)
        x = range(nx)
        x, y = np.meshgrid(x,y)
    elif orientation == 3 :
        # -x,y
        y = range(ny)
        x = range(nx-2,-1,-1)
        x.append(0)
        x, y = np.meshgrid(x,y)
    elif orientation == 4 :
        # -x,-y
        y = range(nx-2,-1,-1)
        y.append(0)
        x = range(nx-2,-1,-1)
        x.append(0)
        x, y = np.meshgrid(x,y)
    elif orientation == 5 :
        # x,y
        y = range(ny)
        x = range(nx)
        y, x = np.meshgrid(x,y)
    elif orientation == 6 :
        # x,-y
        y = range(ny-2,-1,-1)
        y.append(0)
        x = range(nx)
        y, x = np.meshgrid(x,y)
    elif orientation == 7 :
        # -x,y
        y = range(ny)
        x = range(nx-2,-1,-1)
        x.append(0)
        y, x = np.meshgrid(x,y)
    elif orientation == 8 :
        # -x,-y
        y = range(nx-2,-1,-1)
        y.append(0)
        x = range(nx-2,-1,-1)
        x.append(0)
        y, x = np.meshgrid(x,y)
    else :
        print 'orientation must be an integer between 1 and 8.'
    return np.copy(arrayin[y,x])
    
def lena(shape=(256, 256)):
    """Load an image of lena and return the array.

    it is an 512x512 np.uint8 array.
    obtained from: http://www.ece.rice.edu/~wakin/images/
    """
    from scipy import misc
    fnam = os.path.dirname(os.path.realpath(__file__)) + os.path.normcase('/lena512.bmp')
    array = misc.imread(fnam).astype(np.float)
    if shape != (512, 512) :
        array = interpolate(array, shape)
    return array 

def brog(shape=(256, 256)):
    """Load an image of debroglie and return the array.

    it is an 256x256 np.float64 array."""
    fnam = os.path.dirname(os.path.realpath(__file__)) + os.path.normcase('/brog_256x256_32bit_big.raw')
    array = binary_in(fnam, (256,256), dt=np.float32)
    if shape != (256, 256) :
        array = interpolate(array, shape)
    return array 

def twain(shape=(256, 256)):
    """Load an image of Mark Twain and return the array.

    it is an 256x256 np.float64 array."""
    fnam = os.path.dirname(os.path.realpath(__file__)) + os.path.normcase('/twain_256x256_32bit_big.raw')
    array = binary_in(fnam, (256,256), dt=np.float32)
    if shape != (256, 256) :
        array = interpolate(array, shape)
    return array 

def overlap(array1,array2,thresh=0.05e0):
    """Return a mask which is true when array1 * array2 > thresh * max(array1*array2)."""
    arrayout = array1 * array2
    thresh2  = np.max(np.abs(arrayout))*thresh
    arrayout = np.array(1.0 * (np.abs(arrayout) > thresh2),dtype=np.bool)
    return arrayout

def highpass(arrayin, rad=0.5):
    """Perform a highpass on arrayin. rad is the radius of the circle (as a fraction of
    the array size) which is used to filter the inverse Fourier transform of arrayin."""
    circ     = circle(arrayin.shape[0], arrayin.shape[1], radius = rad/2.0)
    arrayout = fft2(ifft2(arrayin) * (1.0 - circ))
    arrayout = np.array(arrayout, dtype=arrayin.dtype)
    return arrayout

def lowpass(arrayin, rad=0.5):
    """Perform a highpass on arrayin. rad is the radius of the circle (as a fraction of
    the array size) which is used to filter the inverse Fourier transform of arrayin."""
    circ     = circle(arrayin.shape[0], arrayin.shape[1], radius = rad/2.0)
    arrayout = fft2(ifft2(arrayin) * circ)
    arrayout = np.array(arrayout, dtype=arrayin.dtype)
    return arrayout

def phase_ramp(N = (1024, 1024), n = (10, 5), origin = 'centre'):
    """Make e^(-2 pi i (nx n + ny m) / N)"""
    if (len(N) == 2) and (len(n) == 2):
        x, y     = make_xy(N, origin = origin)
        exp      = np.exp(-2.0J * np.pi * (n[1] * x / float(N[1]) + n[0] * y / float(N[0]))) 
    elif (len(N) == 1) and (len(n) == 1) :
        x     = make_xy(N, origin = origin)
        exp   = np.exp(-2.0J * np.pi * (n[0] * x / float(N[0]))) 
    elif (len(N) == 2) and (len(n) == 1):
        # ramp the last dimension
        x     = make_xy([N[1]], origin = origin)
        exp   = np.zeros((N[0], N[1]), dtype=np.complex128)
        exp[:]= np.exp(-2.0J * np.pi * (n[0] * x / float(N[1]))) 
    return exp

def Fresnelprop(arrayin, z, wavelength, deltaq):
    """Fresnel propagates arrayin a distance z via a convolution with the Fresnel exponential:
    
    arrayout = arrayin \convolve e^[ i pi r^2 / lambda z ]
             = inverseF[ F[arrayin] x e^[ -i pi lambda z q^2 ] ]
    This works best when z is small."""
    
    N        = arrayin.shape[0]
    x, y     = make_xy(N)
    exp      = np.exp(-1.0J * np.pi * wavelength * z * deltaq**2 * (x**2 + y**2))
    arrayout = fft2(arrayin) * exp
    arrayout = ifft2(arrayout)
    return arrayout

def gaus_Fourier(N, sigma = 1.0):
    """Returns the Gaussian :
        e^[ -(pi sigma q)^2 ] 
    This is a fourier sqace representation such that its inverse fourier transform is normalised.
    sigma is the one e value in units of pixels for the real-space array."""

    x, y   = make_xy(N)
    x      = np.array(x, dtype=np.float64) / np.float(N)
    y      = np.array(y, dtype=np.float64) / np.float(N)
    exp    = np.exp( -(np.pi * sigma)**2 * (x**2 + y**2))
    return exp

def conv_gaus(array, sigma = 1.0):
    """Convolves the array with a gaussian with a 1 / e value of sigma. Returns a complex array."""
    arrayout = fft2(array + 0.0J)
    arrayout = ifft2(arrayout * gaus_Fourier(array.shape[0], sigma))
    arrayout = np.array(arrayout, dtype=array.dtype)
    return arrayout

def rotate(arrayin, phi = 0.0, order = 1):
    """phi in degrees"""
    if phi == 0 :
        return arrayin
    elif arrayin.dtype == 'complex':
        amp      = rotate(np.abs(arrayin), phi, order)
        phase    = rotate(np.angle(arrayin), phi, order)
        arrayout = amp * np.exp(1J * phase)
    else :
        from scipy_hacs import rotate_scipy
        #arrayout = ndimage.interpolation.rotate(arrayin, phi, order = 1, reshape=False, axes=(-2,-1))
        arrayout = rotate_scipy(arrayin, phi, order = order, reshape=False, axes=(-2,-1))
    return arrayout 

def radial_average(array):
    """Radially average the input array. Returns two numpy arrays in a list representing the x,y values of the line plot."""

    def window(xx, yy, x, dx):
        y = 0.0
        count = 0
        for i in range(len(xx)):
            if xx[i] < (x+dx) and xx[i] > (x-dx):
                y     += yy[i]
                count += 1
        return y/float(count)
    ny = array.shape[0]
    nx = ny
     
    x, y = make_xy(ny)
    r    = np.sqrt(x**2 + y**2) 

    Nmax = 100000
    if ny * nx < Nmax :
        Nmax = ny * nx

    rand = np.array(np.random.rand(Nmax) * ny * nx , dtype=np.int)

    plot_r = []
    plot_r.append(r.flatten()[rand])
    plot_r.append(array.flatten()[rand])

    N_window      = 1000
    plot_window_r = []
    plot_window_r.append(np.linspace(np.min(plot_r[0]), np.max(plot_r[0]), N_window))
    dr            = 3.0 * (plot_window_r[0][1] -  plot_window_r[0][0])

    y = []
    for j in range(N_window):
        y.append( window(plot_r[0], plot_r[1], plot_window_r[0][j], dr) )
    plot_window_r.append(np.array(y))
    return plot_window_r

def print_h5(g, offset = '\t\t'):
    """Prints the structure of a h5 file (eg. g = h5py.File('/blah.h5', 'r') )"""
    import h5py
    if   isinstance(g,h5py.File) :
        print g.file, '(File)', g.name

    elif isinstance(g,h5py.Dataset) :
        print '(Dataset)', g.name, '    len =', g.shape #, g.dtype

    elif isinstance(g,h5py.Group) :
        print '(Group)', g.name

    if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
        for key,val in dict(g).iteritems() :
            subg = val
            print offset, key, #,"   ", subg.name #, val, subg.len(), type(subg),
            print_h5(subg, offset + '    ')

def ravel(arrayin):
    """Unpack array into 1d real array [np.ravel(np.real(arrayin)), np.ravel(np.imag(arrayin))]"""
    if arrayin.dtype == complex :
        arrayout = np.concatenate((np.ravel(np.real(arrayin)), np.ravel(np.imag(arrayin))), axis = 0)
    else :
        arrayout = np.ravel(arrayin)
    return arrayout

def unravel(arrayin, shape = 0):
    """Repack arrayin from a bg.ravel array with shape (or assume square complex array)"""
    N = arrayin.shape[0]
    n = int(np.sqrt(N/2))
    if type(shape) != tuple :
        arrayout = arrayin[:N/2].reshape(n,n) + 1.0J * arrayin[N/2:].reshape(n,n)
    else :
        arrayout = arrayin[:N/2].reshape(shape) + 1.0J * arrayin[N/2:].reshape(shape)
    return arrayout

def matrify(xmask, func, xmask_imag = None, bmask = None, dt = np.float64, progress = False):
    """Turn a linear mapping (func) from x to b into a real matrix equation: A . x = b
    
    xmask determines the dimensions of the vector x (optional arguement to exclude imaginary components of x with xmask)
    Anm = Anl . delta(l - m) ( we evaluate this via func ).
    """
    # I need to work out the dimensions of the A matrix
    # xvect  = np.flatten(xarray)[np.flatnonzero(xmask)]
    # xarray[np.nonzero(xmask)] = xvect
    if xmask == None :
        xmask_imag = xmask
    
    if bmask == None :
        barray = func(xmask)
        bmask = np.ones_like(barray, dtype=np.bool)

    class xthing(object):
        def __init__(self, mask, mask_imag = None):
            if mask_imag == None :
                self.mask  = np.vstack((mask,mask))
            else :
                self.mask  = np.vstack((mask,mask_imag))
        def array(self, vect):
            arrayout = np.zeros_like(self.mask, dtype = vect.dtype)
            arrayout[np.nonzero(self.mask)] = vect
            arrayout = np.vsplit(arrayout, 2)
            return (arrayout[0] + 1.0J * arrayout[1])
        def vect(self, array):
            vectout = np.ravel(np.vstack((array.real, array.imag)))[np.flatnonzero(self.mask)]
            return vectout

    x = xthing(xmask, xmask_imag)
    b = xthing(bmask)
    N = 2*np.count_nonzero(bmask)
    M = np.count_nonzero(xmask) + np.count_nonzero(xmask_imag)

    A      = np.zeros((N,M), dtype = dt)
    xvect  = np.zeros((M), dtype=np.bool)

    if progress :
        count = 0
        sys.stdout.write("100- :")
    for m in range(M):
        if progress :
            #print m , '/', M
            count += 1
            if float(count)/float(M) > 0.01 :
                sys.stdout.write("-")
                sys.stdout.flush()
                count = 0
        xvect.fill(0)
        xvect[m] = 1
        A[:,m] = b.vect(func(x.array(xvect)))
    if progress :
        sys.stdout.write("\n")
    return A, x, b

def funcify_3d(arrayin, func2d):
    """If len(arrayin.shape) == 4 or 3 or 2 then apply func2d to the last 2 dimensions of arrayin and return the result in the same shape as arrayin."""
    assert(len(arrayin.shape) >= 2)
    elem = arrayin.size / (arrayin.shape[-1] * arrayin.shape[-2])
    if elem == 2 :
        arrayout = func2d(arrayin)
    else :
        array = arrayin.flatten().reshape( (elem, arrayin.shape[-2], arrayin.shape[-1]))
        arrayout = []
        for i in range(elem):
            arrayout.append(func2d(array[i]))
        arrayout = np.array(arrayout).reshape( arrayin.shape )
    return arrayout
    

class svd(object):
    def __init__(self,A, b):
        self.A = A
        self.b = b
        self.U, self.S, self.Vt = np.linalg.svd(A)
        self.modes     = self.modes_meth()
        self.modes_sum = self.modes_sum_meth()
        self.x = self.x_meth()

    def condition_number_meth(self):
        return self.S[-1]/self.S[0]

    def x_meth(self):
        return np.sum(self.modes_meth(), axis = 0)

    def modes_meth(self):
        w             = np.dot(np.transpose(self.U), self.b)[:self.S.shape[0]] / self.S
        modes         = np.zeros(self.Vt.shape, dtype = self.Vt.dtype)
        for i in range(self.Vt.shape[0]):
            modes[i,:] = np.transpose(self.Vt)[:,i] * w[i]
        return modes

    def modes_sum_meth(self):
        modes_sum         = np.zeros(self.Vt.shape, dtype = self.Vt.dtype)
        for i in range(self.modes.shape[0]):
            modes_sum[i,:] = np.sum(self.modes[:i+1,:], axis = 0)
        return modes_sum

class cgls(object):
    """Do an ILRUFT calculation in general given the functions Adot and ATdot and the bvector.
    
    Solves A . x = b 
    given routines for A . x' and AT . b'
    and the bvector
    where x and b may be any numpy arrays."""

    def __init__(self, Adot, ATdot, bvect):
        self.Adot  = Adot
        self.ATdot = ATdot
        self.bvect = bvect
        self.iterationsILRUFT = 0
        self.error_residual = []

    def ilruft(self, iterations = 10, refresh = 'no'):
        """Iteratively solve the linear equations using the conjugate gradient method.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if refresh == 'yes' :
            self.d     = np.copy(self.bvect)
            self.r     = self.ATdot(self.bvect)
            self.p     = np.copy(self.r)
            self.t     = self.Adot(self.p)
            self.norm_residual  = np.sum(np.abs(self.bvect)**2)

        if self.iterationsILRUFT == 0 :
            self.Sample = self.ATdot(self.bvect)
            self.Sample.fill(0)
            self.d     = np.copy(self.bvect)
            self.r     = self.ATdot(self.bvect)
            self.p     = np.copy(self.r)
            self.t     = self.Adot(self.p)
            self.norm_residual  = np.sum(np.abs(self.bvect)**2)
            self.error_residual = []
        
        for i in range(iterations):
            temp        = np.sum(np.abs(self.r)**2)
            self.alpha  = temp / np.sum(np.abs(self.t)**2)
            self.Sample += self.alpha * self.p
            self.d     -= self.alpha * self.t
            self.r      = self.ATdot(self.d)
            self.betta  = np.sum(np.abs(self.r)**2) / temp
            self.p      = self.r + self.betta * self.p
            self.t      = self.Adot(self.p)
            self.error_residual.append(np.sum(np.abs(self.d)**2)/self.norm_residual)
            #print 'residual error =', self.error_residual[-1]
            if self.error_residual[-1] <= 1.0e-30 :
                break

        self.iterationsILRUFT += iterations
        return self.Sample

class test_retrievals(object):
    def __init__(self):
        """Initialise yo."""
        self.image       = None
        self.Illum       = None
        self.Sample      = None
        self.Exit        = None
        self.sample_area = None
        self.N           = None

        import os
        print os.path.realpath(__file__)

    def STEM(self, N=256):
        """Make a test STEM case."""
        import andrew.STEM as st
        # make a STEM probe that takes up about half the array
        self.stem        = st.STEMprobe()
        self.stem.N      = N
        self.stem.energy = 300.0e3
        self.stem.ampF   = circle(N,N)
        self.stem.Q      = 8.0e0
        self.stem.aberrations['C1'] = 150.0e0
        self.stem.makeParams()
        self.stem.makePhase()
        self.Illum  = self.stem.probeR

        # make brog and twain 
        M           = int(0.23 * float(N)/ 2.0)
        amp         = scale(brog(M) , np.min(np.abs(self.Illum)), np.max(np.abs(self.Illum)))
        phase       = scale(twain(M), -np.pi       , np.pi)
        sample      = amp * np.exp(1.0J * phase)
        sample_area = np.ones((M,M),np.float64)
        self.Sample                                   = np.zeros((N,N),dtype=np.complex128)
        self.sample_area                              = np.zeros((N,N),dtype=np.float64)
        self.Sample[N/2+1:N/2+1+M,N/2+1:N/2+1+M]      = sample
        self.sample_area[N/2+1:N/2+1+M,N/2+1:N/2+1+M] = sample_area
        
        self.Exit  = self.Sample + self.Illum 
        self.image = np.square(np.abs(fft2(self.Exit)))
        self.N     = N
        print "STEM done"

    def pencil(self, N=256):
        """Make a test pencil case."""
        # make a pencil probe that takes up about half the array
        array       = circle(N,N,radius=0.23)
        self.Illum  = array * np.exp(1.0J * array)
        # make brog and twain 
        M           = int(0.23 * float(N)/ 2.0)
        amp         = scale(brog(M) , np.min(array), np.max(array))
        phase       = scale(twain(M), -np.pi       , np.pi)
        sample      = amp * np.exp(1.0J * phase)
        sample_area = np.ones((M,M),np.float64)
        self.Sample                                   = np.zeros((N,N),dtype=np.complex128)
        self.sample_area                              = np.zeros((N,N),dtype=np.float64)
        self.Sample[N/2+1:N/2+1+M,N/2+1:N/2+1+M]      = sample
        self.sample_area[N/2+1:N/2+1+M,N/2+1:N/2+1+M] = sample_area

        self.Exit  = self.Sample + self.Illum 
        self.image = np.square(np.abs(fft2(self.Exit)))
        self.N     = N

    def pencil_transmissive(self, N=256):
        """Make a test pencil case."""
        # make a pencil probe that takes up about half the array
        array       = circle(N,N,radius=0.23)
        self.Illum  = array * np.exp(1.0J * array)
        # make brog and twain 
        M           = int(0.23 * float(N)/ 2.0)
        amp         = scale(brog(M) , np.min(array), np.max(array))
        phase       = scale(twain(M), -np.pi       , np.pi)
        sample      = amp * np.exp(1.0J * phase)
        sample_area = np.ones((M,M),np.float64)
        self.Sample                                   = np.zeros((N,N),dtype=np.complex128)
        self.sample_area                              = np.zeros((N,N),dtype=np.float64)
        self.Sample[N/2+1:N/2+1+M,N/2+1:N/2+1+M]      = sample
        self.sample_area[N/2+1:N/2+1+M,N/2+1:N/2+1+M] = sample_area

        self.Exit  = (self.Sample + 1.0) * self.Illum 
        self.image = np.square(np.abs(fft2(self.Exit)))
        self.N     = N

    def pencil_PILRUFT(self, N=256):
        """Make a test pencil case."""
        # make a pencil probe that takes up about half the array
        array       = circle(N,N,radius=0.23)
        self.Illum  = array * np.exp(1.0J * array)
        # make brog and twain 
        M           = int(0.23 * float(N)/ 2.0)
        amp         = scale(brog(M) , np.min(array), np.max(array))
        phase       = scale(twain(M), -np.pi       , np.pi)
        sample      = amp * np.exp(1.0J * phase)
        sample_area = np.ones((M,M),np.float64)
        self.Sample                                   = np.zeros((N,N),dtype=np.complex128)
        self.sample_area                              = np.zeros((N,N),dtype=np.float64)
        self.Sample[N/2+1:N/2+1+M,N/2+1:N/2+1+M]      = sample
        self.sample_area[N/2+1:N/2+1+M,N/2+1:N/2+1+M] = sample_area

        self.Exit  = (self.Sample + 1.0) * self.Illum 
        self.image = np.square(np.abs(fft2(self.Exit)))
        self.N     = N
