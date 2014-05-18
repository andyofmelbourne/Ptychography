import numpy as np
#import matplotlib.pyplot as plt
from scipy import ndimage
#from matplotlib.colors import LogNorm
#import mahotas
from scipy import fftpack
#import pymorph
from random import *
import scipy as sp
import random
import time
import threading
import sys
import scipy.misc as sm
import os
from ctypes import *

#mkl = cdll.LoadLibrary("mk2_rt.dll")
mkl = cdll.LoadLibrary("libmkl_rt.so")
#c_double_p = POINTER(c_double)
#DFTI_COMPLEX = c_int(32)
#DFTI_DOUBLE = c_int(36)


# Input/Output Routines
def binary_in(fnam, ny = 0, nx = 0, dt = np.dtype(np.float32), endianness='big', dimFnam = False):
    """Read a 2-d array from a binary file."""
    if dimFnam :
        onlyfiles  = [ f for f in os.listdir('.') if os.path.isfile(f) ]
        fnam_match = [ f for f in onlyfiles if f[:len(fnam)] == fnam ]
        dim        = fnam_match[0][len(fnam) + 1 :-4]
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
        arrayout = np.fromfile(fnam_in,dtype=dt).reshape( dim )
    else :
        arrayout = np.fromfile(fnam,dtype=dt).reshape( (ny,nx) )
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    #arrayout = np.float64(arrayout)
    return arrayout

def binary_in_complex(fnam,ny,nx,dt=np.dtype(np.complex128),endianness='big'):
    """Read a 2-d array from a binary file."""
    arrayout = np.fromfile(fnam,dtype=dt).reshape( (ny,nx) )
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    arrayout = np.complex128(arrayout)
    return arrayout

def binary_out(array, fnam, dt=np.dtype(np.float32), endianness='big', appendDim=False):
    """Write a 2-d array to a binary file."""
    if appendDim == True :
        fnam_out = fnam + '_'
        for i in array.shape[:-1] :
            fnam_out += str(i) + 'x' 
        fnam_out += str(array.shape[-1]) + '.raw'
    else :
        fnam_out = fnam
    arrayout = np.array(array,dtype=dt)
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    arrayout.tofile(fnam_out)

def binary_in_amp_phase(fnamAmp,fnamPhase,N,dt=np.float64,endianness='big'):
    """call in fnamAmp and fnamPhase and return the complex array."""
    amp   = binary_in(fnamAmp  ,N,N,dt=dt,endianness=endianness)
    phase = binary_in(fnamPhase,N,N,dt=dt,endianness=endianness)
    return amp * np.exp(1.0J * phase)

def gauss(arrayin,a,ryc=0.0,rxc=0.0): 
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




def fft2_inplace(a):
    desc_handle = c_void_p()
    mkl.DftiCreateDescriptor(byref(desc_handle), c_longlong(36), c_longlong(32), c_longlong(2), (c_longlong*2)(a.shape[0], a.shape[1]))
    mkl.DftiCommitDescriptor(desc_handle)
    mkl.DftiComputeForward(desc_handle, a.ctypes.data_as(c_void_p))
    mkl.DftiFreeDescriptor(desc_handle)
    return a

def ifft2_inplace(a):
    desc_handle = c_void_p()
    mkl.DftiCreateDescriptor(byref(desc_handle), c_longlong(36), c_longlong(32), c_longlong(2), (c_longlong*2)(a.shape[0], a.shape[1]))
    mkl.DftiCommitDescriptor(desc_handle)
    mkl.DftiComputeBackward(desc_handle, a.ctypes.data_as(c_void_p))
    mkl.DftiFreeDescriptor(desc_handle)
    return a

def fft2_memoryLeak(arrayin):
    """Calculate the 2d fourier transform of an array with N/2 - 1 as the zero-pixel."""
    # do an fft
    # check if arrayin is 2d
    a = arrayin.shape
    if len(a) == 1 :
        arrayout = fft(arrayin)
    elif len(a) == 2 :
        ny = arrayin.shape[0]
        nx = arrayin.shape[1]
        arrayout = np.array(arrayin,dtype=np.complex128)
        arrayout = np.roll(arrayout,-(ny/2-1),0)
        arrayout = np.roll(arrayout,-(nx/2-1),1)
        fft2_inplace(arrayout) 
        arrayout/= np.sqrt(float(ny*nx))
        arrayout = np.roll(arrayout,-(ny/2+1),0)
        arrayout = np.roll(arrayout,-(nx/2+1),1)
    elif len(a) == 3 :
        arrayout = np.array(arrayin,dtype=np.complex128)
        for i in range(a[0]):
            arrayout[i] = fft2(arrayin[i])
    return arrayout

def ifft2_memoryLeak(arrayin):
    """Calculate the 2d inverse fourier transform of an array with N/2 - 1 as the zero-pixel."""
    # do an fft
    # check if arrayin is 2d
    a = arrayin.shape
    if len(a) == 1 :
        arrayout = ifft(arrayin)
    elif len(a) == 2 :
        ny = arrayin.shape[0]
        nx = arrayin.shape[1]
        arrayout = np.array(arrayin,dtype=np.complex128)
        arrayout = np.roll(arrayout,-(ny/2-1),0)
        arrayout = np.roll(arrayout,-(nx/2-1),1)
        ifft2_inplace(arrayout) 
        arrayout /= np.sqrt(float(ny*nx))
        arrayout = np.roll(arrayout,-(ny/2+1),0)
        arrayout = np.roll(arrayout,-(nx/2+1),1)
    elif len(a) == 3 :
        arrayout = np.array(arrayin,dtype=np.complex128)
        for i in range(a[0]):
            arrayout[i] = ifft2(arrayin[i])
    return arrayout




def fft2(arrayin):
    """Calculate the 2d fourier transform of an array with N/2 - 1 as the zero-pixel."""
    # do an fft
    # check if arrayin is 2d
    a = arrayin.shape
    if len(a) == 1 :
        arrayout = fft(arrayin)
    elif len(a) == 2 :
        ny = arrayin.shape[0]
        nx = arrayin.shape[1]
        arrayout = np.array(arrayin,dtype=np.complex128)
        arrayout = np.roll(arrayout,-(ny/2-1),0)
        arrayout = np.roll(arrayout,-(nx/2-1),1)
        arrayout = fftpack.fft2(arrayout) / np.sqrt(float(ny*nx))
        arrayout = np.roll(arrayout,-(ny/2+1),0)
        arrayout = np.roll(arrayout,-(nx/2+1),1)
    elif len(a) == 3 :
        arrayout = np.array(arrayin,dtype=np.complex128)
        for i in range(a[0]):
            arrayout[i] = fft2(arrayin[i])
    return arrayout

def ifft2(arrayin):
    """Calculate the 2d inverse fourier transform of an array with N/2 - 1 as the zero-pixel."""
    # do an fft
    # check if arrayin is 2d
    a = arrayin.shape
    if len(a) == 1 :
        arrayout = ifft(arrayin)
    elif len(a) == 2 :
        ny = arrayin.shape[0]
        nx = arrayin.shape[1]
        arrayout = np.array(arrayin,dtype=np.complex128)
        arrayout = np.roll(arrayout,-(ny/2-1),0)
        arrayout = np.roll(arrayout,-(nx/2-1),1)
        arrayout = fftpack.ifft2(arrayout) * np.sqrt(float(ny*nx))
        arrayout = np.roll(arrayout,-(ny/2+1),0)
        arrayout = np.roll(arrayout,-(nx/2+1),1)
    elif len(a) == 3 :
        arrayout = np.array(arrayin,dtype=np.complex128)
        for i in range(a[0]):
            arrayout[i] = ifft2(arrayin[i])
    return arrayout

def fft(arrayin):
    """Calculate the 1d fourier transform of an array with N/2 - 1 as the zero-pixel."""
    # do an fft
    ny = arrayin.shape[0]
    arrayout = np.array(arrayin,dtype=np.complex128)
    arrayout = np.roll(arrayout,-(ny/2-1),0)
    arrayout = fftpack.fft(arrayout) / np.sqrt(float(ny))
    arrayout = np.roll(arrayout,-(ny/2+1),0)
    return arrayout

def ifft(arrayin):
    """Calculate the 2d inverse fourier transform of an array with N/2 - 1 as the zero-pixel."""
    # do an fft
    ny = arrayin.shape[0]
    arrayout = np.array(arrayin,dtype=np.complex128)
    arrayout = np.roll(arrayout,-(ny/2-1),0)
    arrayout = fftpack.ifft(arrayout) * np.sqrt(float(ny))
    arrayout = np.roll(arrayout,-(ny/2+1),0)
    return arrayout

def show(arrayin):
    array = np.abs(arrayin)
    plt.clf()
    plt.ion()
    plt.imshow(array,cmap='hot')
    plt.axis('off')
    plt.show()

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
    if origin is None :
        origin_x, origin_y = nx // 2 - 1, ny // 2 - 1
    else :
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

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

def quadshift(arrayin):
    ny = arrayin.shape[0]
    nx = arrayin.shape[1]
    arrayout = np.roll(arrayin,-(ny/2+1),0)
    arrayout = np.roll(arrayout,-(nx/2+1),1)
    return arrayout

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

def padd(arrayin,ny,nx=None):
    """Padd arrayin with zeros until it is an ny*nx array, keeping the zero pixel at N/2-1
    
    only works when arrayin and arrayout have even domains."""
    if nx == None :
        nx = ny
    nyo = arrayin.shape[0]
    nxo = arrayin.shape[1]
    arrayout = np.zeros((ny,nx),dtype=arrayin.dtype)
    arrayout[(ny-nyo)//2 : (ny-nyo)//2 + nyo,(nx-nxo)//2 : (nx-nxo)//2 + nxo] = arrayin
    return arrayout

def unpadd(arrayin,ny,nx=None):
    """unpadd arrayin with zeros until it is an ny*nx array, keeping the zero pixel at N/2-1
    
    only works when arrayin and arrayout have even domains."""
    if nx == None :
        nx = ny
    nyo = arrayin.shape[0]
    nxo = arrayin.shape[1]
    arrayout = np.zeros((ny,nx),dtype=arrayin.dtype)
    arrayout = arrayin[(nyo-ny)//2 : (nyo-ny)//2 + ny,(nxo-nx)//2 : (nxo-nx)//2 + nx]
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
    
def roll(arrayin,dy = 0,dx = 0):
    """np.roll arrayin by dy in dim 0 and dx in dim 1."""
    if (dy != 0) or (dx != 0):
        arrayout = np.roll(arrayin,dy,0)
        arrayout = np.roll(arrayout,dx,1)
    else :
        arrayout = arrayin
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
        arrayout *= phase_ramp(arrayout.shape[0], y - ny_centre, x - nx_centre)
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

def interpolate(arrayin,new_res):
    """Interpolate to the grid."""
    if arrayin.dtype == 'complex' :
        Ln = interpolate(np.real(arrayin),new_res) + 1.0J * interpolate(np.imag(arrayin),new_res)
        #Ln = interpolate(np.abs(arrayin),new_res) * np.exp(1.0J * interpolate(np.angle(arrayin),new_res))
    else :
        coeffs    = ndimage.spline_filter(arrayin)
        rows,cols = arrayin.shape
        coords    = np.mgrid[0:rows-1:1j*new_res,0:cols-1:1j*new_res]
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
    
def brog(N=256):
    """Load an image of debroglie and return the array.

    it is an 256x256 np.float64 array."""
    import os
    fnam = os.path.dirname(os.path.realpath(__file__)) + os.path.normcase('/test_cases/brog_256x256_32bit_big.raw')
    array = binary_in(fnam,256,256)
    if N != 256 :
        array = interpolate(array,N)
    return array 

def twain(N=256):
    """Load an image of Mark Twain and return the array.

    it is an 256x256 np.float64 array."""
    import os
    fnam = os.path.dirname(os.path.realpath(__file__)) + os.path.normcase('/test_cases/twain_256x256_32bit_big.raw')
    array = binary_in(fnam,256,256)
    if N != 256 :
        array = interpolate(array,N)
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

def phase_ramp(N, ny, nx):
    """Make e^(-2 pi i (nx n + ny m) / N)"""
    x, y     = make_xy(N)
    exp      = np.exp(-2.0J * np.pi * (nx * x + ny * y) / float(N))
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

def rotate(arrayin, phi=90.0):
    """Rotate arrayin by phi degrees. returns an array of the same dimensions."""
    if arrayin.dtype == 'complex' :
        arrayout = rotate(np.abs(arrayin), phi=phi) * np.exp(1.0J * rotate(np.angle(arrayin), phi=phi))
    else :
        arrayout = sp.ndimage.interpolation.rotate(arrayin,phi,reshape=False)
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

# This is a simple job runner for the program SAAF_STEM_basics_x64.exe 
# it will edit the input text file, run the program then grab the desired output

# I want to make the job just act like a module

def saaf_func(unitcells = 1, unitcell_x = 8, Nsupercell = 512, phaseObject = False, slices_perunit = 4.0):
    sa = SAAF(unitcells_z = unitcells, Nsupercell = Nsupercell, unitcell_x = unitcell_x, phaseObject = phaseObject, slices_perunit = slices_perunit )
    out = {}
    out['STEM_images'] = sa.arrays
    out['proj_pot']    = sa.proj_potential_real + 1.0J * sa.proj_potential_imag 
    out['dmap']        = sa.dmap
    out['probe']       = sa.probe_real + 1.0J * sa.probe_imag
    return out

class SAAF(object):
    def __init__(self, unitcells_z = 1, Nsupercell = 256, unitcell_x = 5, probePoints = 0, phaseObject = False, slices_perunit = 4.0):
        # Set directories
        self.execution_dir = 'C:\\Users\\andyofmelbourne\Desktop\\for_Andrew\\SAAF_STEM_basics\\'
        self.exe_file      = 'SAAF_STEM_basics.exe'
        self.input_file    = 'blank.txt'
        self.user_input    = 'user_input.txt'
        self.output_dir    = 'output_test/'
        
        self.unit_cell_thickness = 3.90500000000000
        self.options       = self.setVariables()
        ###########################################
        # over ride defaults
        crystal_thickness = ['Crystal thickness', self.unit_cell_thickness * unitcells_z, 20]
        self.options['crystal_thickness'] = crystal_thickness 

        supercell_x    = ['Number of pixels in x-dirn', Nsupercell, 5]
        self.options['supercell_x'] = supercell_x    
        #
        supercell_y    = ['Number of pixels in y-dirn', Nsupercell, 6]
        self.options['supercell_y'] = supercell_y    
        #
        unitcells_x    = ['Number of unit cells in x-dirn', unitcell_x, 7]
        self.options['unitcells_x'] = unitcells_x    
        #
        unitcells_y    = ['Number of unit cells in y-dirn', unitcell_x, 8]
        self.options['unitcells_y'] = unitcells_y    
        #
        if probePoints == 0 :
            probe_points_x = ['Number of probe points in x-dirn', Nsupercell / unitcell_x, 46]
            self.options['probe_points_x'] = probe_points_x 
            probe_points_y = ['Number of probe points in y-dirn', Nsupercell / unitcell_x, 47]
            self.options['probe_points_y'] = probe_points_y 
        #
        if phaseObject:
            full_prop      = ['<1> full propagation, <2> POA', 2, 52]
            self.options['full_prop'] = full_prop      
        #
        slice_thickness   = ['Slice thickness', self.unit_cell_thickness / slices_perunit, 21]
        self.options['slice_thickness'] = slice_thickness   
        #
        self.write_options(self.options, self.input_file)
        self.run_job()
        self.arrays = self.grab_files()


    def order_options(self, options):
        opList = []
        ii = 0
        for i in range(len(options)):
            for item in options.values():
                if ii == item[-1]:
                    opList.append(item[:-1])
                    ii = ii + 1
        return opList

    def grab_files(self):
        """Grab the output of the program and put them into arrays for processing."""
        onlyfiles = [ f for f in os.listdir(self.execution_dir) if os.path.isfile(os.path.join(self.execution_dir,f)) ]
        
        # grab the projected potentials
        fnam_real = [f for f in onlyfiles if self.options['output_file_real'][1] == f ]
        fnam_imag = [f for f in onlyfiles if self.options['output_file_imag'][1] == f ]
        
        if len(fnam_real) > 0 :
            self.proj_potential_real = np.loadtxt(os.path.join(self.execution_dir, fnam_real[0]))
            os.remove(os.path.join(self.execution_dir, fnam_real[0]))
        if len(fnam_imag) > 0 :
            self.proj_potential_imag = np.loadtxt(os.path.join(self.execution_dir, fnam_imag[0]))
            os.remove(os.path.join(self.execution_dir, fnam_imag[0]))

        # grab the real space STEM probe
        if self.options['probe_output'][1] == 1 :
            fnam_probe_real = [f for f in onlyfiles if 'probe_real.txt' == f ]
            fnam_probe_imag = [f for f in onlyfiles if 'probe_imag.txt' == f ]
        

        if (len(fnam_probe_real) > 0) & (len(fnam_probe_imag) > 0) :
            self.probe_real = np.loadtxt(os.path.join(self.execution_dir, fnam_probe_real[0]))
            os.remove(os.path.join(self.execution_dir, fnam_probe_real[0]))
            self.probe_imag = np.loadtxt(os.path.join(self.execution_dir, fnam_probe_imag[0]))
            os.remove(os.path.join(self.execution_dir, fnam_probe_imag[0]))

        # grab every file starting with fnam basis
        only_basis = [f for f in onlyfiles if self.options['fnam_basis'][1] == f[:len(self.options['fnam_basis'][1])] ]
        # there is an det_info file I want to get rid of
        only_arrays   = [f for f in only_basis if  f.find('det_info') == -1 ]

        only_det_info = [f for f in only_basis if  f.find('det_info') != -1 ]
        if len(only_det_info) > 0:
            with open(os.path.join(self.execution_dir, only_det_info[0]), "r") as myfile:
                self.det_info = myfile.readlines()
            os.remove(os.path.join(self.execution_dir, only_det_info[0]))
        
        self.dmap = np.loadtxt(os.path.join(self.execution_dir, self.options['dmap_fnam'][1]))
        os.remove(os.path.join(self.execution_dir, self.options['dmap_fnam'][1]))
        
        # read in the arrays
        arrays = []
        for i in range(len(only_arrays)):
            arrays.append(np.loadtxt(os.path.join(self.execution_dir, only_arrays[i])))
            os.remove(os.path.join(self.execution_dir, only_arrays[i]))
        return arrays

    def run_job(self):
        # execute program
        # get the current working directory
        cwd = os.getcwd()
        # set the current working directory to the execution directory
        os.chdir(self.execution_dir)
        # execute the program
        os.system(self.exe_file)
        # return to the directory of this program
        os.chdir(cwd)

    def write_options(self, options, input_file):
        """Write a list of string value pairs to the file input_file in the directory self.execution_dir"""
        # get the current working directory
        cwd = os.getcwd()
        # set the current working directory to the execution directory
        os.chdir(self.execution_dir)
        # write to file
        f = open(input_file, 'w')
        for option in self.order_options(options):
            f.write(option[0] + '\n')
            f.write(str(option[1]) + '\n')
            print option[0] + '\n'
            print str(option[1]) + '\n'
        f.close()
        # return to the directory of this program
        os.chdir(cwd)
        return 

    def setVariables(self):
        """A long list of variables used to construct the input file for executable.
        """
        options = {}
        numThreads     = ['', 0, 0] # use all threads
        options['numThreads'] = numThreads     
        absorptive_Fph = ['<1> absorptive, <2> FPh, <0> exit', 1, 1]
        options['absorptive_Fph'] = absorptive_Fph 
        MS_abs_menu    = ['MS abs menu', 1, 2]
        options['MS_abs_menu'] = MS_abs_menu    
        xtl_fnam       = ['.xtl file name', 'srtio3_001_200.xtl', 3]
        options['xtl_fnam'] = xtl_fnam       
        zone_sysRow    = ['<1> zone axis, <2> sys row', 1, 4]
        options['zone_sysRow'] = zone_sysRow    
        supercell_x    = ['Number of pixels in x-dirn', 256, 5]
        options['supercell_x'] = supercell_x    
        supercell_y    = ['Number of pixels in y-dirn', 256, 6]
        options['supercell_y'] = supercell_y    
        unitcells_x    = ['Number of unit cells in x-dirn', 3, 7]
        options['unitcells_x'] = unitcells_x    
        unitcells_y    = ['Number of unit cells in y-dirn', 3, 8]
        options['unitcells_y'] = unitcells_y    
        contin         = ['<1> continue, <2> change', 1, 9]
        options['contin'] = contin         
        absorption     = ['<1> include absorption, <2> otherwise', 1, 10]
        options['absorption'] = absorption     
        absorp_calc    = ['absorptive potential: <1> calc, <2> read', 1, 11]
        options['absorp_calc'] = absorp_calc    
        save_formfact  = ['Inelastic form factors: <1> save, <2> otherwise', 2, 12]
        options['save_formfact'] = save_formfact  
        MS_abs_menu2   = ['MS abs menu', 2, 13]
        options['MS_abs_menu2'] = MS_abs_menu2   
        D1D            = ['<1> 2D output, <2> 1D output', 1, 14]
        options['2D1D'] = D1D
        D1D2           = ['<1> 2D output, <2> 1D output', 1, 15]
        options['2D1D2'] = D1D2
        realImag       = ['Output: <1> Re+Im, <2> Re, <3> Im', 1, 16]
        options['realImag'] = realImag
        output_file    = ['Output file', 'proj_potential_real.txt', 17]
        options['output_file_real'] = output_file
        output_file2   = ['Output file', 'proj_potential_imag.txt', 18]
        options['output_file_imag'] = output_file2
        MS_abs_menu3   = ['MS abs menu', 5, 19]
        options['MS_abs_menu3'] = MS_abs_menu3   
        crystal_thickness = ['Crystal thickness', 3.90500000000000, 20]
        options['crystal_thickness'] = crystal_thickness 
        slice_thickness   = ['Slice thickness', self.unit_cell_thickness / 4.0, 21]
        options['slice_thickness'] = slice_thickness   
        contin2        = ['<1> continue, <2> change', 1, 22]
        options['contin2'] = contin2        
        BWL            = ['<1> BWL, <2> otherwise', 1, 23]
        options['BWL'] = BWL            
        rings          = ['Number of rings in polar dirn', 2, 24]
        options['rings'] = rings          
        ring_divns     = ['Number of divisions in azimuthal dirn', 4, 25]
        options['ring_divns'] = ring_divns     
        ang_offset     = ['Angular offset for azimuthal integration', 45.0000000000000, 26]
        options['ang_offset'] = ang_offset     
        confirm        = ['<1> confirm, <2> change', 1, 27]
        options['confirm'] = confirm        
        theta_inner    = ['Theta inner angle', 0.00000000000000, 28]
        options['theta_inner'] = theta_inner    
        theta_outer    = ['Theta outer angle', 11.500000, 29]
        options['theta_outer'] = theta_outer    
        theta_inner2   = ['Theta inner angle', 11.500000, 30]
        options['theta_inner2'] = theta_inner2   
        theta_outer2   = ['Theta outer angle', 23.000000, 31]
        options['theta_outer2'] = theta_outer2    
        detector_shift = ['<1> shift detector off axis, <2> otherwise', 2, 32]
        options['detector_shift'] = detector_shift 
        pixel_suff     = ['<1> pixel sufficiency test, <2> otherwise', 1, 33]
        options['pixel_suff'] = pixel_suff     
        safety_factor  = ['Safety factor', 3, 34]
        options['safety_factor'] = safety_factor  
        output_detector_map = ['<1> output detector config, <2> otherwise', 1, 35]
        options['output_detector_map'] = output_detector_map 
        dmap_fnam      = ['File name for detector config', 'detector_map.txt', 36]
        options['dmap_fnam'] = dmap_fnam      
        bf_detector    = ['<1> include BF detector, <2> otherwise', 2, 37]
        options['bf_detector'] = bf_detector    
        Cs             = ['Cs in mm', 0.00000000000000, 38]
        options['Cs'] = Cs             
        Scherzer       = ['<1> use Scherzer defocus, <2> change', 1, 39]
        options['Scherzer'] = Scherzer       
        confirm2       = ['<1> use this cutoff, <2> change', 2, 40]
        options['confirm2'] = confirm2       
        aperture_size  = ['New cutoff', 0.917000000000000, 41] # this is in A401
        options['aperture_size'] = aperture_size  
        confirm3       = ['<1> use this cutoff, <2> change', 1, 42]
        options['confirm3'] = confirm3       
        include_C5     = ['<1> include C5, <2> otherwise', 2, 43]
        options['include_C5'] = include_C5     
        contin3        = ['<1> continue, <2> change', 1, 44]
        options['contin3'] = contin3        
        confirm4       = ['<1> OK, <2> change size, <3> change direction', 1, 45]
        options['confirm4'] = confirm4       
        probe_points_x = ['Number of probe points in x-dirn', 64, 46]
        options['probe_points_x'] = probe_points_x 
        probe_points_y = ['Number of probe points in y-dirn', 64, 47]
        options['probe_points_y'] = probe_points_y 
        probe_shift    = ['<1> shift incident probe, <2> otherwise', 2, 48]
        options['probe_shift'] = probe_shift    
        probe_output   = ['<1> output probe, <2> dont', 1, 49]
        options['probe_output'] = probe_output    
        fnam_basis     = ['File name basis', 'output_diffs', 50]
        options['fnam_basis'] = fnam_basis     
        output_separate= ['Output: <1> single, <2> separate, <3> both', 2, 51]
        options['output_separate'] = output_separate
        full_prop      = ['<1> full propagation, <2> POA', 1, 52]
        options['full_prop'] = full_prop      
        MS_abs         = ['MS abs menu', 0, 53]
        options['MS_abs'] = MS_abs         
        contin4        = ['<1> absorptive, <2> FPh, <0> exit', 0, 54]
        options['contin4'] = contin4        
        return options
