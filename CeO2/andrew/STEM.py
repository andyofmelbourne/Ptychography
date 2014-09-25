import numpy as np
import matplotlib.pyplot as plt
import bagOfns as bg

class STEMprobe(object):
    """A STEM probe."""

    def __init__(self):
        """Initialise the class variables."""
        self.lamb       = None
        self.energy     = None
        self.k          = None

        self.N          = None
        self.dq         = None
        self.Q          = None
        self.dx         = None
        self.X          = None
        self.drad       = None
        self.Rad        = None

        self._ampF       = None
        self._phaseF     = None
        self._probeF     = None

        self._phaseFwrapped = None

        self._ampR       = None
        self._phaseR     = None
        self._probeR     = None

        self.aberrations= {}
    
    def makePhase(self):
        """Make the Phase at the detector."""
        N = self.N
        self.phaseF = np.zeros((N,N))
        
        A0 = A1 = A2 = A3 = A4 = A5 = B2 = B4 = C1 = C3 = C5 = D4 = S3 = 0.0 + 0.0J

        if self.aberrations.has_key('A0'): A0 = self.aberrations['A0']
        if self.aberrations.has_key('A1'): A1 = self.aberrations['A1']
        if self.aberrations.has_key('A2'): A2 = self.aberrations['A2']
        if self.aberrations.has_key('A3'): A3 = self.aberrations['A3']
        if self.aberrations.has_key('A4'): A4 = self.aberrations['A4']
        if self.aberrations.has_key('A5'): A5 = self.aberrations['A5']
        if self.aberrations.has_key('B2'): B2 = self.aberrations['B2']
        if self.aberrations.has_key('B4'): B4 = self.aberrations['B4']
        if self.aberrations.has_key('C1'): C1 = self.aberrations['C1']
        if self.aberrations.has_key('C3'): C3 = self.aberrations['C3']
        if self.aberrations.has_key('C5'): C5 = self.aberrations['C5']
        if self.aberrations.has_key('D4'): D4 = self.aberrations['D4']
        if self.aberrations.has_key('S3'): S3 = self.aberrations['S3']

        array = np.zeros((N,N),dtype=np.complex128)
        wq    = np.zeros((N,N),dtype=np.complex128)

        for ii in range(N):
            for jj in range(N):
                i = ii - N/2 + 1
                j = jj - N/2 + 1
                wq[ii,jj] = self.drad * (np.float(i) + 1.0J * np.float(j))
                
        if np.abs(A0) > 0.0 :
            array += A0*np.conj(wq)

        if np.abs(A1) > 0.0 :
            array += A1*np.conj(wq)**2/2.0

        if np.abs(A2) > 0.0 :
            array += A2*np.conj(wq)**3/3.0

        if np.abs(A3) > 0.0 :
            array += A3*np.conj(wq)**4/4.0

        if np.abs(A4) > 0.0 :
            array += A4*np.conj(wq)**5/5.0

        if np.abs(A5) > 0.0 :
            array += A5*np.conj(wq)**6/6.0

        if np.abs(B2) > 0.0 :
            array += B2*wq**2*np.conj(wq)

        if np.abs(B4) > 0.0 :
            array += B4*wq**3*np.conj(wq)**2

        if np.abs(C1) > 0.0 :
            array += C1*wq*np.conj(wq)/2.0

        if np.abs(C3) > 0.0 :
            array += C3*(wq*np.conj(wq))**2/4.0

        if np.abs(C5) > 0.0 :
            array += C5*(wq*np.conj(wq))**3/6.0

        if np.abs(D4) > 0.0 :
            array += D4*wq**4*np.conj(wq)

        if np.abs(S3) > 0.0 :
            array += S3*wq**3*np.conj(wq)

        self.phaseF = -2.0 * np.pi * np.real(array) / self.lamb

    def addC1(self,C1):
        """Add defocus aberrations to the current phase."""
        N = self.N
        array = np.zeros((N,N),dtype=np.complex128)
        for ii in range(N):
            for jj in range(N):
                i = ii - N/2 + 1
                j = jj - N/2 + 1
                wq = self.drad * (i + 1.0J * j)
                
                array[ii,jj] += C1*wq*np.conj(wq)/2.0

        self.phaseF += 2.0 * np.pi * np.real(array) / self.lamb

    def makeParams(self):
        """Determine parameters in terms of the ones provided."""
        if self.N == None:
            raise NameError('need to define the side length of the array (N)')

        if self.lamb != None:
            self.k      = 1.0 / self.lamb
            self.energy = bg.energyK(self.k)

        elif self.k != None:
            self.lamb   = 1.0 / self.k
            self.energy = bg.energyK(self.k)

        elif self.energy != None:
            self.k      = bg.waveno(self.energy)
            self.lamb   = 1.0 / self.k

        else : 
            raise NameError('need to define the wavelength,enery or wavenumber of the electron (lamb,energy,k)')

        if self.dq != None :
            self.Q    = self.N * self.dq
            self.dx   = 1.0/self.Q
            self.X    = self.N * self.dx 
            self.drad = self.lamb * self.dq
            self.Rad  = self.N * self.drad / 2.0

        elif self.Q != None :
            self.dq   = self.Q / self.N
            self.dx   = 1.0/self.Q
            self.X    = self.N * self.dx 
            self.drad = self.lamb * self.dq
            self.Rad  = self.N * self.drad / 2.0

        elif self.dx != None :
            self.X    = self.N * self.dx 
            self.dq   = 1.0/self.X
            self.Q    = self.dq * self.N
            self.drad = self.lamb * self.dq
            self.Rad  = self.N * self.drad / 2.0

        elif self.X != None :
            self.dx   = self.X / self.N 
            self.dq   = 1.0/self.X
            self.Q    = self.dq * self.N
            self.drad = self.lamb * self.dq
            self.Rad  = self.N * self.drad / 2.0

        elif self.drad != None :
            self.Rad  = self.N * self.drad / 2.0
            self.dq   = self.drad / self.lamb
            self.Q    = self.dq * self.N
            self.dx   = 1.0/self.Q
            self.X    = self.N * self.dx 

        elif self.Rad != None :
            self.drad = 2.0 * self.Rad / self.N
            self.dq   = self.drad / self.lamb
            self.Q    = self.dq * self.N
            self.dx   = 1.0/self.Q
            self.X    = self.N * self.dx 

        else :
            raise NameError('need to define q,x or phi')

    def disParams(self):
        """Display the class Variables."""
        print 50 * '-'
        print 'array size \t\t',self.N
        print 'Energy     \t\t',self.energy 
        print 'wavenumber \t\t',self.k      
        print 'lamb       \t\t',self.lamb 
        print 'X          \t\t',self.X    
        print 'dela x     \t\t',self.dx
        print 'Q          \t\t',self.Q
        print 'dela q     \t\t',self.dq
        print 'Rads       \t\t',self.Rad
        print 'dela rads  \t\t',self.drad
        print 50 * '-'

    def phasePlate(self):
        """Show the phase plate for the probe."""
        from numpy import random

        array    = np.zeros((self.N,self.N),dtype=np.complex128)
        arrayout = np.zeros((self.N,self.N),dtype=np.complex128)

        arrayout = random.random_sample((self.N,self.N)) + 1.0J*random.random_sample((self.N,self.N))
        
        array = bg.ifft2(self.app * np.exp(1.0J*self.phaseApp))

        arrayout *= array
        arrayout = bg.fft2(arrayout)

        return np.abs(arrayout)

    def defocusSweep(self,dmin=-100.0e0,dmax=100.0e0,n=101):
        """Use this to make one of those side profile graphs of the probe as a function of defocus.  """
        profile = np.zeros((n, self.N), dtype=np.complex128)
        dlist   = np.linspace(dmin,dmax,n)
        for i in range(n) :
            print dlist[i], dmax
            self.aberrations['C1'] = dlist[i]
            self.makePhase()
            profile[i,:] = self.ampR[self.N/2 - 1, :]
        return profile

    def dradFromAperture(self, aperture, halfAngle):
        """Returns the radians per pixel from the number of radians between the centre and the edge of the aperture."""
        # first I will threshold the aperture to give us an array of 1's and 0's
        mask = 1 * (aperture > 0.3 * np.max(aperture))
        # find the maximum diameter alonge the 0 axis
        # this is done by projecting along the 1 axis and looking at the domain
        line  = np.sum(mask, axis = 0)
        line  = 1 * (line >= 1)
        diam1 = np.sum(line)

        line  = np.sum(mask, axis = 1)
        line  = 1 * (line >= 1)
        diam2 = np.sum(line)

        diam = np.float(diam1 + diam2) / 2.0
        drad = 2.0 * halfAngle / diam
        return drad

    @property
    def probeF(self):
        if (self._probeF != None):
            return self._probeF
        elif self._ampF != None and self._phaseF != None :
            return (self._ampF * np.exp(1.0J * self._phaseF))
        else :
            # this will be None if we can't calculate it but that's the point
            return bg.fft2(self.probeR)
    @probeF.setter
    def probeF(self, value):
        self._probeF = value
        self._ampF   = None
        self._phaseF = None
        self._ampR   = None
        self._phaseR = None
        self._probeR = None

    @property
    def ampF(self):
        if (self._ampF != None):
            return self._ampF
        elif self._probeF != None :
            return (np.abs(self._probeF))
        elif (self._ampR != None and self._phaseR != None) or (self._probeR != None):
            # this will be None if we can't calculate it but that's the point
            return np.abs(bg.fft2(self.probeR))
        else :
            return None
    @ampF.setter
    def ampF(self, value):
        self._ampF   = value
        self._probeF = None
        self._ampR   = None
        self._phaseR = None
        self._probeR = None

    @property
    def phaseF(self):
        if (self._phaseF != None):
            return self._phaseF
        elif self._probeF != None :
            return (np.angle(self._probeF))
        elif (self._ampR != None and self._phaseR != None) or (self._probeR != None):
            return np.angle(bg.fft2(self.probeR))
        else :
            return None
    @phaseF.setter
    def phaseF(self, value):
        self._phaseF = value
        self._probeF = None
        self._ampR   = None
        self._phaseR = None
        self._probeR = None

    @property
    def probeR(self):
        if (self._probeR != None):
            return self._probeR
        elif self._ampR != None and self._phaseR != None :
            return (self._ampR * np.exp(1.0J * self._phaseR))
        else :
            # this will be None if we can't calculate it but that's the point
            return bg.ifft2(self.probeF)
    @probeF.setter
    def probeF(self, value):
        self._probeR = value
        self._ampR   = None
        self._phaseR = None
        self._ampF   = None
        self._phaseF = None
        self._probeF = None

    @property
    def ampR(self):
        if (self._ampR != None):
            return self._ampR
        elif self._probeR != None :
            return (np.abs(self._probeR))
        else :
            # this will be None if we can't calculate it but that's the point
            return np.abs(bg.ifft2(self.probeF))
    @ampR.setter
    def ampR(self, value):
        self._ampR   = value
        self._probeR = None
        self._ampF   = None
        self._phaseF = None
        self._probeF = None

    @property
    def phaseR(self):
        if (self._phaseR != None):
            return self._phaseR
        elif self._probeR != None :
            return (np.angle(self._probeR))
        else :
            # this will be None if we can't calculate it but that's the point
            return np.angle(bg.ifft2(self.probeF))
    @phaseR.setter
    def phaseR(self, value):
        self._phaseR = value
        self._probeR = None
        self._ampF   = None
        self._phaseF = None
        self._probeF = None
    
    @property
    def phaseFwrapped(self):
        if self.phaseF == None :
            print 'must calculate phaseF first!'
            return None
        else :
            array = np.exp(1.0J * self.phaseF)
            array = np.angle(array)
            return array


