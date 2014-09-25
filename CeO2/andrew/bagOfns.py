import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import LogNorm
import mahotas
from scipy import fftpack
from random import *
import scipy as sp
import random
import time
import threading
import sys
import scipy.misc as sm

# Input/Output Routines
def binary_in(fnam,ny,nx,dt=np.dtype(np.float32),endianness='big'):
    """Read a 2-d array from a binary file."""
    arrayout = np.fromfile(fnam,dtype=dt).reshape( (ny,nx) )
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    arrayout = np.float64(arrayout)
    return arrayout

def binary_in_complex(fnam,ny,nx,dt=np.dtype(np.complex128),endianness='big'):
    """Read a 2-d array from a binary file."""
    arrayout = np.fromfile(fnam,dtype=dt).reshape( (ny,nx) )
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    arrayout = np.complex128(arrayout)
    return arrayout

def binary_out(array,fnam,dt=np.dtype(np.float32),endianness='big'):
    """Write a 2-d array to a binary file."""
    arrayout = np.array(array,dtype=dt)
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    arrayout.tofile(fnam)

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
    """Tile the 2d array into a 1d array via map."""
    n = map.shape[0] 
    arrayout = np.zeros((2*n),dtype=np.float64)
    for ii in range(n):
        i = map[ii,0]
        j = map[ii,1]
        arrayout[ii]   = array[i,j].real
        arrayout[ii+n] = array[i,j].imag
    return arrayout

def unmapping(array, map,N):
    """Untile the 1d array into a 2d array via map."""
    n = map.shape[0] 
    arrayout = np.zeros((N,N),dtype=np.complex128)
    for ii in range(n):
        i = map[ii,0]
        j = map[ii,1]
        arrayout[i,j]   = array[ii] + 1.0J * array[ii+n]
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
    """Calculate sum |array1 - array2|^2 / sum|array1|^2 ."""
    tot = np.sum(np.abs(array1)**2)
    return np.sum(np.abs(array1-array2)**2)/tot

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
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = N, N
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
    """Crop arrayin to the smallest rectangle that contains all of the non-zero elements and return the result. If mask is given use that to determine non-zero elements."""
    if mask==None :
        mask = arrayin
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
    return arrayin[top:bottom+1,left:right+1]
    
def roll(arrayin,dy = 0,dx = 0):
    """np.roll arrayin by dy in dim 0 and dx in dim 0."""
    if (dy != 0) or (dx != 0):
        arrayout = np.roll(arrayin,dy,0)
        arrayout = np.roll(arrayout,dx,1)
    else :
        arrayout = arrayin
    return arrayout

def roll_to(arrayin,y = 0,x = 0):
    """np.roll arrayin such that the centre of the array is moved to y and x.
    
    Where the centre is N/2 - 1"""
    ny, nx = np.shape(arrayin)
    arrayout = roll(arrayin, dy = y - ny/2 + 1,dx = x - nx/2 + 1)
    return arrayout

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


