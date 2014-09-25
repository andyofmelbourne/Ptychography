import numpy as np
import bagOfns as bg

class phaseThing(object):
    """A class for the phase problem."""

    def __init__(self,N = None, illumAmpR = None, illumPhaseR = None, sampleAmp = None, samplePhase = None, sampleArea = None, Illum_R = None, sample_area = None, Sample_known = None, image = None):
        """initialise class variables."""
        self.N                  = N 
        self.illum_amp_R        = None
        self.illum_phase_R      = None
        self.Illum_R            = Illum_R
        
        self.sample_area        = sample_area
        
        self.Sample             = None 
        self.Exit               = None 
        
        self.sample_amp_known   = None
        self.sample_phase_known = None 
        self.Sample_known       = Sample_known
        
        self._Exit_known        = None
        
        self.image_amp          = None
        
        self._Autoc             = None
        self._Cross             = None
        self._Cross_area        = None
        self._Autoc_illum       = None
        self._Autoc_illum_area  = None
        self._Autoc_sample_area = None
        
        if image != None :
            image = image * (image > 0.0)
            self.image_amp = np.sqrt(image)

        if self.N != None :
            self.Sample = np.zeros((self.N,self.N), dtype = np.complex128)
        
        if illumAmpR != None :
            self.illum_amp_R       = bg.binary_in(illumAmpR,N,N,dt=np.float64,endianness='big')
        
        if illumPhaseR != None :
            self.illum_phase_R     = bg.binary_in(illumPhaseR,N,N,dt=np.float64,endianness='big')
            
        if sampleAmp != None :
            self.sample_amp_known  = bg.binary_in(sampleAmp,N,N,dt=np.float64,endianness='big')
        
        if samplePhase != None :
            self.sample_phase_known= bg.binary_in(samplePhase,N,N,dt=np.float64,endianness='big')
        
        if sampleArea != None :
            self.sample_area = bg.binary_in(sampleArea,N,N,dt=np.float64,endianness='big')
        
        if (self.illum_amp_R != None) and (self.illum_phase_R != None):
            self.Illum_R = self.illum_amp_R * np.exp(1.0J * self.illum_phase_R)
        
        if (self.sample_amp_known != None) and (self.sample_phase_known != None):
            self.Sample_known = self.sample_amp_known * np.exp(1.0J * self.sample_phase_known)
        
    def dis_sample(self,array=None,mask='array'):
        """Display the current sample function without the padded zero's.

        If array != None then do this for array instead of sample."""
        if array != None and mask=='array':
            tempAC = array
        elif mask == 'sampleArea':
            tempAC = self.sample_area
        else :
            tempAC = self.Sample
        #most left point 
        for i in range(self.N):
            tot = np.sum(np.abs(tempAC[:,i]))
            if tot > 0.0 :
                break
        left = i
        #most right point 
        for i in range(self.N-1,-1,-1):
            tot = np.sum(np.abs(tempAC[:,i]))
            if tot > 0.0 :
                break
        right = i
        #most up point 
        for i in range(self.N):
            tot = np.sum(np.abs(tempAC[i,:]))
            if tot > 0.0 :
                break
        top = i
        #most down point
        for i in range(self.N-1,-1,-1):
            tot = np.sum(np.abs(tempAC[i,:]))
            if tot > 0.0 :
                break
        bottom = i
        if array != None :
            tempAC = array
        else :
            tempAC = self.Sample
        return tempAC[top:bottom+1,left:right+1]

    def normalise_image(self,tot=1.0):
        """Normalise the image so that sum image_amp**2 = tot * sum |Illum_R|**2 ."""
        tot2 = tot * np.sum(np.real( self.Illum_R * np.conj(self.Illum_R) ))
        self.image_amp = bg.normalise(self.image_amp, tot = tot2)

    def Pmod(self,psi=None):
        """Apply the modulus constraint to the sample function."""
        if psi != None :
            tempAC = psi
        else :
            tempAC = self.Sample
        tempAC = bg.fft2(tempAC + self.Illum_R)
        tempAC = self.image_amp * np.exp(1.0J * np.angle(tempAC))
        tempAC = bg.ifft2(tempAC) - self.Illum_R
        return tempAC

    def Pmod_exit(self,psi=None):
        """Apply the modulus constraint to the exit wave."""
        if psi != None :
            tempAC = psi
        else :
            tempAC = self.Exit
        tempAC = bg.fft2(tempAC)
        tempAC = self.image_amp * np.exp(1.0J * np.angle(tempAC))
        tempAC = bg.ifft2(tempAC)
        return tempAC

    def Psub(self,psi=None):
        """Apply the support constraint to the sample function."""
        if psi != None :
            tempAC = psi
        else :
            tempAC = self.Sample
        tempAC *= self.sample_area
        return tempAC

    def ERA(self, psi = None, iters = 1):
        """Do (iters) iterations of the Error Reduction Algorithm (ERA).
        
        This applies the modulus and sample constraints to the sample function
        not the exit wave.""" 
        if psi == None :
            psi = self.Sample
        arrayout = np.copy(psi)
        for i in range(iters):
            arrayout = self.Pmod(arrayout)
            arrayout = self.Psub(arrayout)
        return arrayout 

    def errorMod(self,psi = None):
        """Calculate the mean squared error between the retrieved amplitude and 
        the square root of the detected intensity at the detector.
        
        The error metric is normalised with respect to the detected intensity."""
        if psi == None :
            psi = self.Sample
        array = self.image_amp - np.abs(bg.fft2(psi + self.Illum_R)) 
        error = np.sum(np.square(array))/np.sum(np.square(self.image_amp))
        return error

    def errorMixed(self,psi = None):
        """Returns sum|Psub(Sample) - Pmod(Sample)|^2 / sum|Sample|^2."""
        if psi == None :
            psi = self.Sample
        a = np.sum(np.abs(psi)**2)
        if a != 0.0e0 :
            error = np.sum(np.abs(self.Psub(psi) - self.Pmod(psi))**2)/a
        else :
            error = 0.0e0
        return error

    def errorSup():
        pass

    ########## Properties ##########
    @property
    def Exit_known(self):
        return self._Exit_known
    @Exit_known.setter
    def Exit_known(self, value):
        self._Exit_known = value
        self.image_amp   = np.abs(bg.fft2(self._Exit_known))

    @property
    def Autoc(self):
        if self.image_amp == None :
            print 'must calculate image_amp first!'
            return None
        else :
            array = bg.ifft2(np.square(self.image_amp))
            return array

    @property
    def Cross(self):
        if self.Illum_R == None or self.image_amp == None :
            print 'must calculate Illum_R and image_amp first!'
            return None
        else :
            array = bg.ifft2(np.square(self.image_amp) - np.square(np.abs(bg.fft2(self.Illum_R))))
            return array

    @property
    def Cross_area(self):
        if self.Illum_R == None or self.image_amp == None :
            print 'must calculate Illum_R and image_amp first!'
            return None
        else :
            array = bg.ifft2(np.square(self.image_amp) - np.square(np.abs(bg.fft2(self.Illum_R))))
            array = np.array(np.abs(array),dtype=np.float64)
            tol = np.max(array) * 1.0e-8
            array = 1.0 * (array > tol)
            return array

    @property
    def Autoc_illum(self):
        if self.Illum_R == None :
            print 'must calculate Illum_R first!'
            return None
        else :
            array = bg.ifft2(np.square(np.abs(bg.fft2(self.Illum_R))))
            return array

    @property
    def Autoc_illum_area(self):
        if self.Illum_R == None :
            print 'must calculate Illum_R and image_amp first!'
            return None
        else :
            array = bg.ifft2(np.square(np.abs(bg.fft2(self.Illum_R))))
            array = np.array(np.abs(array),dtype=np.float64)
            tol = np.max(array) * 1.0e-8
            array = 1.0 * (array > tol)
            return array

    @property
    def Autoc_sample_area(self):
        if self.sample_area == None :
            print 'must calculate sample_area first!'
            return None
        else :
            array = bg.ifft2(np.square(np.abs(bg.fft2(self.sample_area + 0.0J))))
            array = np.array(np.abs(array),dtype=np.float64)
            tol = np.max(array) * 1.0e-8
            array = 1.0 * (array > tol)
            return array

class linearProb(phaseThing):
    """A sub class."""

    def __init__(self,N = None, illumAmpR = None, illumPhaseR = None, sampleAmp = None, samplePhase = None, sampleArea = None, Illum_R = None, sample_area = None, Sample_known = None, image = None):
        phaseThing.__init__(self,N, illumAmpR, illumPhaseR, sampleAmp, samplePhase, sampleArea, Illum_R, sample_area, Sample_known, image)
        self.xvect              = None
        self.bvect              = None
        self.Amatrix            = None
        self.neqvect            = None
        self.nvarvect           = None
        self.singular_values    = None
        self.residual           = None
        self.rank               = None
        self.condition_number   = None
        self.iterationsILRUFT   = 0
        self._Sample2           = None
        self._Sample_div         = None
        
    def setVects(self):
        """Set up the (1d) xvect and bvect for the A . x = b linear equations."""
        nvarvect = np.zeros((self.N**2,2),dtype = int)
        neqvect  = np.zeros((self.N**2,2),dtype = int)
        nvar = 0
        for i in range(self.N):
            for j in range(self.N):
                if self.sample_area[i,j] > 0.5e0 :
                    nvarvect[nvar,0] = i
                    nvarvect[nvar,1] = j
                    nvar += 1
        
        cross_area   = self.Cross_area
        sample_autoc = self.Autoc_sample_area #np.zeros((self.N, self.N), dtype=np.float64) 
        neq = 0
        for i in range(self.N-1):
            for j in range(self.N/2 - 1):
                if (cross_area[i,j] > 0.5e0) and (sample_autoc[i,j] < 0.5e0) :
                    neqvect[neq,0] = i
                    neqvect[neq,1] = j
                    neq += 1
        
        j = self.N/2 - 1
        for i in range(self.N/2):
            if (cross_area[i,j] > 0.5e0) and (sample_autoc[i,j] < 0.5e0) :
                neqvect[neq,0] = i
                neqvect[neq,1] = j
                neq += 1
        
        self.nvarvect = np.resize(nvarvect,(nvar,2))
        self.neqvect  = np.resize(neqvect ,(neq ,2))
        
        self.bvect = bg.mapping(self.Cross, self.neqvect)
        self.xvect = np.zeros((2*nvar),dtype=np.float64)

    def makeAmatrix(self):
        """Make the Amatrix for the linear retrieval."""
        if self.neqvect == None or self.nvarvect == None :
            self.setVects()
        neqv   = self.neqvect
        neq    = neqv.shape[0]
        nvarv  = self.nvarvect
        nvar   = nvarv.shape[0]
        array1 = np.copy(self.Illum_R)
        array2 = bg.fft2(np.conj(bg.fft2(self.Illum_R)))
        self.Amatrix = np.zeros((2*neq,2*nvar),dtype=np.float64)
        x = np.array(nvarv[:,0])
        y = np.array(nvarv[:,1])

        for i in range(neq):
            array22  = np.roll(array2 , neqv[i,0]-self.N/2+1,0)
            array22  = np.roll(array22, neqv[i,1]-self.N/2+1,1)
            array22 /= np.sqrt(float(self.N)**2)

            array11  = np.roll(array1 ,-neqv[i,0]+self.N/2-1,0)
            array11  = np.roll(array11,-neqv[i,1]+self.N/2-1,1)
            array11 /= np.sqrt(float(self.N)**2)

            array3 = array22[x,y]
            self.Amatrix[i,:nvar]      = array3.real
            self.Amatrix[i,nvar:]      =-array3.imag
            self.Amatrix[i+neq,:nvar]  = array3.imag
            self.Amatrix[i+neq,nvar:]  = array3.real

            array3 = array11[x,y]
            self.Amatrix[i,:nvar]     += array3.real
            self.Amatrix[i,nvar:]     += array3.imag
            self.Amatrix[i+neq,:nvar] += array3.imag
            self.Amatrix[i+neq,nvar:] -= array3.real

    def checkAmatrix(self):
        """Check the Amatrix by calculating ||A.x - b|| with the known sample."""
        if self.Sample_known == None :
            print 'need to calculate the sample function.'
            return None
        if self.nvarvect == None :
            print 'need to calculate the nvarvect.'
            return None
        if self.neqvect == None :
            print 'need to calculate the neqvect.'
            return None
        xvect = bg.mapping(self.Sample_known,self.nvarvect)
        error = sum(np.square(np.dot(self.Amatrix,xvect)-self.bvect))/sum(np.square(self.bvect))
        return error

    def solveSample_lin(self,rcond=-1):
        """Once the Amatrix has been made solve the problem."""
        if self.Amatrix == None or self.bvect == None :
            print "Need to calculate the Amatrix and the b vector."
            return None
        self.xvect, self.residual, self.rank, self.singular_values = np.linalg.lstsq(self.Amatrix,self.bvect,rcond=rcond)
        self.Sample = bg.unmapping(self.xvect,self.nvarvect,self.N)
        self.condition_number = self.singular_values[0] / self.singular_values[self.xvect.shape[0]-1]

    def ilruft(self,iterations):
        """Iteratively solve the linear equations using the conjugate gradient method.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iterationsILRUFT == 0 :
            self.Illum_F = bg.fft2(self.Illum_R)
            self.xvect = 0.0
            self.d     = self.bvect
            self.r     = self.ATdot(self.bvect)
            self.p     = self.r
            self.t     = self.Adot(self.p)
            self.norm_residual  = np.sum(self.bvect**2)
            self.error_residual = []
        
        for i in range(iterations):
            temp        = np.sum(self.r**2)
            self.alpha  = temp / np.sum(self.t**2)
            self.xvect += self.alpha * self.p
            self.d     -= self.alpha * self.t
            self.r      = self.ATdot(self.d)
            self.betta  = np.sum(self.r**2) / temp
            self.p      = self.r + self.betta * self.p
            self.t      = self.Adot(self.p)
            self.error_residual.append(np.sum(self.d**2)/self.norm_residual)
            print 'residual error =', self.error_residual[-1]
        
        self.Sample     = bg.unmapping(self.xvect, self.nvarvect, self.N)
        #self.Sample     = self.ERA()
        self.Exit       = self.Sample + self.Illum_R
        self.iterationsILRUFT += iterations

    def Adot(self, vect):
        """Calculate the matrix vector product A . x using FFTs."""
        tempAC = bg.unmapping(vect, self.nvarvect, self.N)
        tempAC = bg.fft2(tempAC)
        tempAC = 2.0 * np.real(self.Illum_F * np.conj(tempAC)) + 0.0J
        tempAC = bg.ifft2(tempAC)
        return bg.mapping(tempAC, self.neqvect)

    def ATdot(self, vect):
        """Calculate the matrix vector product AT . x using FFTs.
        
        AT means the transpose of A."""
        tempAC = bg.unmapping(vect, self.neqvect, self.N)
        tempAC = bg.fft2(tempAC)
        tempAC = self.Illum_F * np.conj(tempAC) + self.Illum_F * tempAC
        tempAC = bg.ifft2(tempAC)
        return bg.mapping(tempAC, self.nvarvect)

    @property
    def Sample2(self):
        if self.xvect == None or self.nvarvect == None :
            print 'must calculate xvect and nvarvect first!'
            return None
        else :
            array = bg.unmapping(self.xvect, self.nvarvect, self.N)
            return array
    @property
    def Sample_div(self):
        if self.Exit == None or self.Illum_R == None :
            print 'must calculate the exit wave and the illumination first!'
            return None
        else :
            mask   = np.abs(self.Illum_R)
            thresh = 0.3 * np.max(mask)
            mask   = 1.0 * (mask > thresh)
            array  = np.conj(self.Illum_R) * self.Exit / (self.Illum_R * np.conj(self.Illum_R))
            array *= mask
            return array

class linearProb_exit(linearProb):
    """A sub-sub class."""

    def setVects(self):
        """Same as before but with a new b-vector.

        Set up the (1d) xvect and bvect for the A . x = b linear equations."""
        nvarvect = np.zeros((self.N**2,2),dtype = int)
        neqvect  = np.zeros((self.N**2,2),dtype = int)
        nvar = 0
        for i in range(self.N):
            for j in range(self.N):
                if self.sample_area[i,j] > 0.5e0 :
                    nvarvect[nvar,0] = i
                    nvarvect[nvar,1] = j
                    nvar += 1
        
        autoc_illum_area   = self.Autoc_illum_area
        sample_autoc       = self.Autoc_sample_area
        neq = 0
        for i in range(self.N-1):
            for j in range(self.N/2 - 1):
                if (autoc_illum_area[i,j] > 0.5e0) and (sample_autoc[i,j] < 0.5e0) :
                    neqvect[neq,0] = i
                    neqvect[neq,1] = j
                    neq += 1
        
        j = self.N/2 - 1
        for i in range(self.N/2):
            if (autoc_illum_area[i,j] > 0.5e0) and (sample_autoc[i,j] < 0.5e0) :
                neqvect[neq,0] = i
                neqvect[neq,1] = j
                neq += 1
        
        self.nvarvect = np.resize(nvarvect,(nvar,2))
        self.neqvect  = np.resize(neqvect ,(neq ,2))
        
        array      = self.Autoc - bg.autoc(self.Illum_R * (self.sample_area < 0.5e0))
        self.bvect = bg.mapping(array, self.neqvect)
        self.xvect = np.zeros((2*nvar),dtype=np.float64)
    
    def checkAmatrix(self):
        """Check the Amatrix by calculating ||A.x - b|| with the known sample."""
        if self.Sample_known == None :
            print 'need to calculate the sample function.'
            return None
        if self.nvarvect == None :
            print 'need to calculate the nvarvect.'
            return None
        if self.neqvect == None :
            print 'need to calculate the neqvect.'
            return None
        array  = (np.abs(self.Illum_R) > 1.0e-10)
        array2 = np.zeros((self.N,self.N),dtype=np.complex128)
        for ii in range(self.nvarvect.shape[0]):
            i = self.nvarvect[ii,0]
            j = self.nvarvect[ii,1]
            if array[i,j]:
                array2[i,j] = 1.0e0 + self.Sample_known[i,j] * np.conj(self.Illum_R[i,j]) / np.square(np.abs(self.Illum_R[i,j]))

        xvect = bg.mapping(array2,self.nvarvect)
        self.xvect = xvect
        error = sum(np.square(np.dot(self.Amatrix,xvect)-self.bvect))/sum(np.square(self.bvect))
        return error

class linearProb_trans(phaseThing):
    """retrieve the transmissive sample function."""
    
    def __init__(self,N = None, illumAmpR = None, illumPhaseR = None, sampleAmp = None, samplePhase = None, sampleArea = None, Illum_R = None, sample_area = None, Sample_known = None, image = None):
        phaseThing.__init__(self,N, illumAmpR, illumPhaseR, sampleAmp, samplePhase, sampleArea, Illum_R, sample_area, Sample_known, image)
        self.xvect              = None
        self.bvect              = None
        self.neqvect            = None
        self.nvarvect           = None
        self.residual           = None
        self.iterationsILRUFT   = 0
        #self._Sample_div         = None
    
    def setVects(self):
        """Set up the (1d) xvect and bvect for the A . x = b linear equations."""
        nvarvect = np.zeros((self.N**2,2),dtype = int)
        neqvect  = np.zeros((self.N**2,2),dtype = int)
        nvar = 0
        for i in range(self.N):
            for j in range(self.N):
                if self.sample_area[i,j] > 0.5e0 :
                    nvarvect[nvar,0] = i
                    nvarvect[nvar,1] = j
                    nvar += 1
        
        cross_area   = self.Cross_area
        sample_autoc = self.Autoc_sample_area
        neq = 0
        for i in range(self.N-1):
            for j in range(self.N/2 - 1):
                if (cross_area[i,j] > 0.5e0) and (sample_autoc[i,j] < 0.5e0) :
                    neqvect[neq,0] = i
                    neqvect[neq,1] = j
                    neq += 1
        
        j = self.N/2 - 1
        for i in range(self.N/2):
            if (cross_area[i,j] > 0.5e0) and (sample_autoc[i,j] < 0.5e0) :
                neqvect[neq,0] = i
                neqvect[neq,1] = j
                neq += 1
        
        self.nvarvect = np.resize(nvarvect,(nvar,2))
        self.neqvect  = np.resize(neqvect ,(neq ,2))
        
        self.bvect = bg.mapping(self.Cross, self.neqvect)
        self.xvect = np.zeros((2*nvar),dtype=np.float64)

    def ilruft(self,iterations):
        """Iteratively solve the linear equations using the conjugate gradient method.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iterationsILRUFT == 0 :
            self.Illum_F = bg.fft2(self.Illum_R)
            self.xvect = 0.0
            self.d     = np.copy(self.bvect)
            self.r     = self.ATdot(self.bvect)
            self.p     = np.copy(self.r)
            self.t     = self.Adot(self.p)
            self.norm_residual  = np.sum(self.bvect**2)
            self.error_residual = []
        
        for i in range(iterations):
            temp        = np.sum(self.r**2)
            self.alpha  = temp / np.sum(self.t**2)
            self.xvect += self.alpha * self.p
            self.d     -= self.alpha * self.t
            self.r      = self.ATdot(self.d)
            self.betta  = np.sum(self.r**2) / temp
            self.p      = self.r + self.betta * self.p
            self.t      = self.Adot(self.p)
            #self.error_residual.append(np.sum(self.d**2)/self.norm_residual)
            self.error_residual.append(bg.l2norm(self.bvect,self.Adot(self.xvect)))
            print 'residual error =', self.error_residual[-1]
        
        self.Sample     = bg.unmapping(self.xvect, self.nvarvect, self.N)
        self.Sample     = self.ERA()
        self.Exit       = (self.Sample + 1.0) * self.Illum_R
        self.iterationsILRUFT += iterations
    
    def Adot(self, vect):
        """Calculate the matrix vector product A . x using FFTs."""
        tempAC = bg.unmapping(vect, self.nvarvect, self.N)
        tempAC = 2.0 * np.real(self.Illum_F * np.conj(bg.fft2(self.Illum_R*tempAC))) + 0.0J
        tempAC = bg.ifft2(tempAC)
        return bg.mapping(tempAC, self.neqvect)

    def ATdot(self, vect):
        """Calculate the matrix vector product AT . x using FFTs.
        
        AT means the transpose of A."""
        tempAC = bg.unmapping(vect, self.neqvect, self.N)
        tempAC = bg.fft2(tempAC)
        tempAC = self.Illum_F * np.conj(tempAC) + self.Illum_F * tempAC
        tempAC = np.conj(self.Illum_R) * bg.ifft2(tempAC)
        return bg.mapping(tempAC, self.nvarvect)
    
    def unmappSample(self,vect): 
        """Unravel vect into the sample area then make it one outside of the sample area."""
        tempAC = bg.unmapping(vect, self.nvarvect, self.N)
        tempAC += 1.0 - self.sample_area 
        return tempAC
    
    def Pmod(self,psi=None):
        """Apply the modulus constraint to the sample function."""
        if psi != None :
            tempAC = np.copy(psi)
        else :
            tempAC = np.copy(self.Sample)
        tempAC = bg.fft2((1.0 + tempAC) * self.Illum_R)
        tempAC = self.image_amp * np.exp(1.0J * np.angle(tempAC))
        tempAC = bg.ifft2(tempAC)
        tempAC = self.div_by_Illum(array1=tempAC)
        return tempAC
    
    def Psub(self,psi=None):
        """Apply the support constraint to the sample function."""
        if psi != None :
            tempAC = np.copy(psi)
        else :
            tempAC = np.copy(self.Sample)
        tempAC *= self.sample_area
        return tempAC
    
    def ERA(self, psi = None, iters = 1):
        """Do (iters) iterations of the Error Reduction Algorithm (ERA).
        
        This applies the modulus and sample constraints to the sample function
        not the exit wave.""" 
        if psi == None :
            psi = self.Sample
        arrayout = np.copy(psi)
        for i in range(iters):
            arrayout = self.Pmod(arrayout)
            arrayout = self.Psub(arrayout)
        return arrayout 
    
    def div_by_Illum(self, array1=None, array2=None):
        """Do a threshold divide by the illumination function and set the rest to one."""
        if array1 is None :
            array1 = self.Exit
        if array2 is None :
            array2 = self.Illum_R
        mask   = self.sample_area #np.abs(array2)
        thresh = 0.3 * np.max(mask)
        mask   = 1.0 * (mask > thresh)
        array  = (mask > 0.0) * np.conj(array2) * array1 / (array2 * np.conj(array2))
        array  = mask * (array - 1.0)
        return array




class phaseThing_trans(object):
    """A class for the phase problem for single shot CDI assuming Exit = Sample + Illum and a sample support."""
    
    def __init__(self, N, Illum, diff, sampleArea):
        """Initialise the phaseThing_additive variables."""
        self.N          = N
        self.Illum      = Illum
        self.diff       = np.abs(diff)
        self.imageAmp   = np.sqrt(np.abs(diff))
        self.sampleArea = sampleArea
        self.Sample     = np.zeros((N,N), dtype=Illum.dtype)

    def Pmod(self,psi=None):
        """Apply the modulus constraint to the sample function."""
        if psi != None :
            tempAC = np.copy(psi)
        else :
            tempAC = np.copy(self.Sample)
        
        tempAC = bg.fft2((1.0 + tempAC) * self.Illum)
        tempAC = self.imageAmp * np.exp(1.0J * np.angle(tempAC))
        tempAC = bg.ifft2(tempAC) / self.Illum - 1.0
        return tempAC

    def Psub(self,psi=None):
        """Apply the support constraint to the sample function."""
        if psi != None :
            tempAC = psi
        else :
            tempAC = self.Sample
        tempAC *= self.sampleArea
        return tempAC
    
    def ERA(self, psi = None, iters = 1):
        """Do (iters) iterations of the Error Reduction Algorithm (ERA).
        
        This applies the modulus and sample constraints to the sample function
        not the exit wave.""" 
        if psi == None :
            psi = self.Sample
        arrayout = np.copy(psi)
        for i in range(iters):
            arrayout = self.Pmod(arrayout)
            arrayout = self.Psub(arrayout)
        return arrayout 

    def HIO(self, psi = None, iters = 1, beta = 1.0):
        """Do (iters) iterations of the Hybrid Input-Output algorithm (HIO).
        
        This applies the modulus and sample constraints to the sample function
        not the exit wave.""" 
        if psi == None :
            psi = self.Sample
        psi = np.copy(psi)
        for i in range(iters):
            temp      = self.Pmod(psi)
            
            arrayout  = (1.0 + 1.0/beta) * temp 

            arrayout -= psi/beta

            arrayout  = self.Psub(arrayout)

            arrayout -= temp

            psi       = psi + beta * arrayout
        return psi 

    def normalise_image(self,tot=1.0):
        """Normalise the image so that sum imageAmp**2 = tot * sum |Illum|**2 ."""
        tot2           = tot * np.sum(np.real( self.Illum * np.conj(self.Illum) ))
        self.imageAmp  = bg.normalise(self.imageAmp, tot = tot2)
        self.diff      = np.square(self.imageAmp)
    
    def errorMod(self,psi = None):
        """Calculate the mean squared error between the retrieved amplitude and 
        the square root of the detected intensity at the detector.
        
        The error metric is normalised with respect to the detected intensity."""
        if psi == None :
            psi = self.Sample
        array = self.imageAmp - np.abs(bg.fft2((psi + 1.0) * self.Illum)) 
        error = np.sum(np.square(array))/np.sum(self.diff)
        return error

    def errorMixed(self,psi = None):
        """Returns sum|Psub(Sample) - Pmod(Sample)|^2 / sum|Sample|^2."""
        if psi == None :
            psi = self.Sample
        a = np.sum(np.abs(psi)**2)
        if a != 0.0e0 :
            error = np.sum(np.abs(self.Psub(psi) - self.Pmod(psi))**2)/a
        else :
            error = 0.0e0
        return error

    def errorSup():
        pass
    
    def Cross_calc(self):
        """Calculate the cross terms contaminated with the sample autocorrelation terms with F-1 { diff } - autoc{Illum}."""
        array = bg.ifft2(self.diff) - bg.autoc(self.Illum)
        return array
    
    def sampleAutocArea(self):
        """Calculate the domain of the sample's autocorrelation function."""
        array = bg.autoc(self.sampleArea + 0.0J)
        array = bg.blurthresh(array, thresh = 1.0e-8, blur = 0)
        return array

class ILRUFT_trans(phaseThing_trans):
    """A for using ILRUFT when assuming Exit = (1 + Sample) * Illum."""
    
    def __init__(self, N, Illum, diff, sampleArea):
        phaseThing_trans.__init__(self, N, Illum, diff, sampleArea)
        
        self.Illum_F            = bg.fft2(self.Illum)          # it is faster to keep this in memory for later
        self.sampleAutoc        = self.sampleAutocArea()
        self.bvect              = self.Cross_calc() * (1.0 - self.sampleAutoc)
        self.bvectArea          = np.ones((self.N,self.N),dtype=self.Illum.dtype) * (1.0 - self.sampleAutoc)
        self.residual           = None
        self.iterationsILRUFT   = 0
        self.Cross              = self.Cross_calc()
    
    def ilruft_full_area(self,iterations=1):
        """Iteratively solve the linear equations using the conjugate gradient method. 

        This routine assumes that the autocorrelation function of the sample is small compared to 
        the cross terms.
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iterationsILRUFT == 0 :
            self.Sample.fill(0) 
            self.d     = np.copy(self.Cross)
            self.r     = self.ATdot(self.d)
            self.p     = np.copy(self.r)
            self.t     = self.Adot_full(self.p)
            self.norm_residual  = np.sum(np.abs(self.d)**2)
            self.error_residual = []
        
        for i in range(iterations):
            temp        = np.sum(np.abs(self.r)**2)
            self.alpha  = temp / np.sum(np.abs(self.t)**2)
            self.Sample+= self.alpha * self.p
            self.d     -= self.alpha * self.t
            self.r      = self.ATdot(self.d)
            self.betta  = np.sum(np.abs(self.r)**2) / temp
            self.p      = self.r + self.betta * self.p
            self.t      = self.Adot_full(self.p)
            self.error_residual.append(np.sum(np.abs(self.d)**2)/self.norm_residual)
            print 'residual error =', self.error_residual[-1]
        
        self.Exit              = (1.0 + self.Sample) * self.Illum
        self.iterationsILRUFT += iterations

    def ilruft(self,iterations=1):
        """Iteratively solve the linear equations using the conjugate gradient method.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iterationsILRUFT == 0 :
            self.Sample.fill(0) 
            self.d              = np.copy(self.bvect)
            self.r              = self.ATdot(self.bvect)
            self.p              = np.copy(self.r)
            self.t              = self.Adot(self.p)
            self.norm_residual  = np.sum(np.abs(self.bvect)**2)
            self.error_residual = []
        
        for i in range(iterations):
            temp        = np.sum(np.abs(self.r)**2)
            self.alpha  = temp / np.sum(np.abs(self.t)**2)
            self.Sample+= self.alpha * self.p
            self.d     -= self.alpha * self.t
            self.r      = self.ATdot(self.d)
            self.betta  = np.sum(np.abs(self.r)**2) / temp
            self.p      = self.r + self.betta * self.p
            self.t      = self.Adot(self.p)
            self.error_residual.append(np.sum(np.abs(self.d)**2)/self.norm_residual)
            print 'residual error =', self.error_residual[-1]
        
        self.Exit              = (1.0 + self.Sample) * self.Illum
        self.iterationsILRUFT += iterations
    
    def Adot(self, array):
        """Calculate the matrix vector product A . x using FFTs. Follows the algorithm from https://tcmpwiki.ph.unimelb.edu.au/wiki/Transmissive_sample_retrieval"""
        tempAC = 2.0 * np.real(self.Illum_F * np.conj(bg.fft2(self.Illum*array))) + 0.0J
        tempAC = bg.ifft2(tempAC) * self.bvectArea
        return tempAC
    
    def ATdot(self, array):
        """Calculate the matrix vector product AT . x using FFTs. Follows the algorithm from https://tcmpwiki.ph.unimelb.edu.au/wiki/Transmissive_sample_retrieval
        
        AT means the transpose of A."""
        tempAC = bg.fft2(array)
        tempAC = self.Illum_F * np.conj(tempAC) + self.Illum_F * tempAC
        tempAC = np.conj(self.Illum) * bg.ifft2(tempAC) * self.sampleArea
        return tempAC

    def Adot_full(self, array):
        """Calculate the matrix vector product A . x using FFTs. Follows the algorithm from https://tcmpwiki.ph.unimelb.edu.au/wiki/Transmissive_sample_retrieval"""
        tempAC = 2.0 * np.real(self.Illum_F * np.conj(bg.fft2(self.Illum*array))) + 0.0J
        tempAC = bg.ifft2(tempAC) 
        return tempAC




class phaseThing_add(object):
    """A class for the phase problem for single shot CDI assuming Exit = Sample + Illum and a sample support."""
    
    def __init__(self, N, Illum, diff, sampleArea):
        """Initialise the phaseThing_additive variables."""
        self.N          = N
        self.Illum      = Illum
        self.diff       = np.abs(diff)
        self.imageAmp   = np.sqrt(np.abs(diff))
        self.sampleArea = sampleArea
        self.Sample     = np.zeros((N,N), dtype=Illum.dtype)

    def Pmod(self,psi=None):
        """Apply the modulus constraint to the sample function."""
        if psi != None :
            tempAC = np.copy(psi)
        else :
            tempAC = np.copy(self.Sample)
        tempAC = bg.fft2(tempAC + self.Illum)
        tempAC = self.imageAmp * np.exp(1.0J * np.angle(tempAC))
        tempAC = bg.ifft2(tempAC) - self.Illum
        return tempAC

    def Psub(self,psi=None):
        """Apply the support constraint to the sample function."""
        if psi != None :
            tempAC = np.copy(psi)
        else :
            tempAC = np.copy(self.Sample)
        tempAC *= self.sampleArea
        return tempAC
    
    def ERA(self, psi = None, iters = 1):
        """Do (iters) iterations of the Error Reduction Algorithm (ERA).
        
        This applies the modulus and sample constraints to the sample function
        not the exit wave.""" 
        if psi == None :
            psi = self.Sample
        arrayout = np.copy(psi)
        for i in range(iters):
            arrayout = self.Pmod(arrayout)
            arrayout = self.Psub(arrayout)
        return arrayout 

    def HIO(self, psi = None, iters = 1, beta = 1.0):
        """Do (iters) iterations of the Hybrid Input-Output algorithm (HIO).
        
        This applies the modulus and sample constraints to the sample function
        not the exit wave.""" 
        if psi == None :
            psi = self.Sample
        psi = np.copy(psi)
        for i in range(iters):
            temp      = self.Pmod(psi)
            
            arrayout  = (1.0 + 1.0/beta) * temp 

            arrayout -= psi/beta

            arrayout  = self.Psub(arrayout)

            arrayout -= temp

            psi       = psi + beta * arrayout
        return psi 

    def normalise_image(self,tot=1.0):
        """Normalise the image so that sum imageAmp**2 = tot * sum |Illum|**2 ."""
        tot2           = tot * np.sum(np.real( self.Illum * np.conj(self.Illum) ))
        self.imageAmp = bg.normalise(self.imageAmp, tot = tot2)
        self.diff      = np.square(self.imageAmp)
    
    def errorMod(self,psi = None):
        """Calculate the mean squared error between the retrieved amplitude and 
        the square root of the detected intensity at the detector.
        
        The error metric is normalised with respect to the detected intensity."""
        if psi == None :
            psi = self.Sample
        array = self.imageAmp - np.abs(bg.fft2(psi + self.Illum)) 
        error = np.sum(np.square(array))/np.sum(self.diff)
        return error

    def errorMixed(self,psi = None):
        """Returns sum|Psub(Sample) - Pmod(Sample)|^2 / sum|Sample|^2."""
        if psi == None :
            psi = self.Sample
        a = np.sum(np.abs(psi)**2)
        if a != 0.0e0 :
            error = np.sum(np.abs(self.Psub(psi) - self.Pmod(psi))**2)/a
        else :
            error = 0.0e0
        return error

    def errorSup():
        pass
    
    def Cross_calc(self):
        """Calculate the cross terms contaminated with the sample autocorrelation terms with F-1 { diff } - autoc{Illum}."""
        array = bg.ifft2(self.diff) - bg.autoc(self.Illum)
        return array
    
    def sampleAutocArea(self):
        """Calculate the domain of the sample's autocorrelation function."""
        array = bg.autoc(self.sampleArea + 0.0J)
        array = bg.blurthresh(array, thresh = 1.0e-8, blur = 0)
        return array

class ILRUFT_add(phaseThing_add):
    """A for using ILRUFT when assuming Exit = Sample + Illum."""
    
    def __init__(self, N, Illum, diff, sampleArea):
        phaseThing_add.__init__(self, N, Illum, diff, sampleArea)
        
        self.Illum_F            = bg.fft2(Illum)          # it is faster to keep this in memory for later
        self.sampleAutoc        = self.sampleAutocArea()
        self.bvect              = self.Cross_calc() * (1.0 - self.sampleAutoc)
        self.bvectArea          = np.ones((self.N,self.N),dtype=self.Illum.dtype) * (1.0 - self.sampleAutoc)
        self.residual           = None
        self.iterationsILRUFT   = 0
        self.Cross              = self.Cross_calc()
    
    def ilruft_full_area(self,iterations=1):
        """Iteratively solve the linear equations using the conjugate gradient method. 

        This routine assumes that the autocorrelation function of the sample is small compared to 
        the cross terms.
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iterationsILRUFT == 0 :
            self.Sample.fill(0) 
            self.d     = np.copy(self.Cross)
            self.r     = self.ATdot(self.d)
            self.p     = np.copy(self.r)
            self.t     = self.Adot_full(self.p)
            self.norm_residual  = np.sum(np.abs(self.d)**2)
            self.error_residual = []
        
        for i in range(iterations):
            temp        = np.sum(np.abs(self.r)**2)
            self.alpha  = temp / np.sum(np.abs(self.t)**2)
            self.Sample+= self.alpha * self.p
            self.d     -= self.alpha * self.t
            self.r      = self.ATdot(self.d)
            self.betta  = np.sum(np.abs(self.r)**2) / temp
            self.p      = self.r + self.betta * self.p
            self.t      = self.Adot_full(self.p)
            self.error_residual.append(np.sum(np.abs(self.d)**2)/self.norm_residual)
            print 'residual error =', self.error_residual[-1]
        
        self.Exit              = self.Sample + self.Illum
        self.iterationsILRUFT += iterations

    def Adot_full(self, array):
        """Calculate the matrix vector product A . x using FFTs."""
        tempAC = bg.fft2(array)
        tempAC = 2.0 * np.real(self.Illum_F * np.conj(tempAC)) + 0.0J
        tempAC = bg.ifft2(tempAC) 
        return tempAC

    def ilruft(self,iterations=1):
        """Iteratively solve the linear equations using the conjugate gradient method.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iterationsILRUFT == 0 :
            self.Sample.fill(0) 
            self.d              = np.copy(self.bvect)
            self.r              = self.ATdot(self.bvect)
            self.p              = np.copy(self.r)
            self.t              = self.Adot(self.p)
            self.norm_residual  = np.sum(np.abs(self.bvect)**2)
            self.error_residual = []
        
        for i in range(iterations):
            temp        = np.sum(np.abs(self.r)**2)
            self.alpha  = temp / np.sum(np.abs(self.t)**2)
            self.Sample+= self.alpha * self.p
            self.d     -= self.alpha * self.t
            self.r      = self.ATdot(self.d)
            self.betta  = np.sum(np.abs(self.r)**2) / temp
            self.p      = self.r + self.betta * self.p
            self.t      = self.Adot(self.p)
            self.error_residual.append(np.sum(np.abs(self.d)**2)/self.norm_residual)
            print 'residual error =', self.error_residual[-1]
        
        self.Exit              = self.Sample + self.Illum
        self.iterationsILRUFT += iterations
    
    def Adot(self, array):
        """Calculate the matrix vector product A . x using FFTs."""
        tempAC = bg.fft2(array)
        tempAC = 2.0 * np.real(self.Illum_F * np.conj(tempAC)) + 0.0J
        tempAC = bg.ifft2(tempAC) * self.bvectArea
        return tempAC
    
    def ATdot(self, array):
        """Calculate the matrix vector product AT . x using FFTs.
        
        AT means the transpose of A."""
        tempAC = bg.fft2(array)
        tempAC = self.Illum_F * np.conj(tempAC) + self.Illum_F * tempAC
        tempAC = bg.ifft2(tempAC) * self.sampleArea
        return tempAC





# we will have a list of sample areas 
# and also the combined sample area (the sub-set of the full sample area that is defined in terms linear equations)
# a list of cross terms (and the combined list) 1d and 2d
class PILRUFT(object):
    """Implement the Ptychographic Iterative Linear Retrieval Using Fourier Transforms algorithm."""

    def __init__(self, N, Illum, diffs, sampleArea, posis):
        """Initialise the PILRUFT variables."""
        self.N               = N          # the side length of the detector
        self.Illum           = Illum      # the complex illumination function
        self.diffs           = diffs      # an array of numpy arrays
        self.posis           = posis      # the list of probe positions
        self.Nimages         = len(diffs)
        #---> There are three sample areas 
        #       - The total size of the object
        #       - The overlap of the illumination with 
        #         the total sample areas for each probe 
        #         position
        #       - The complete area of the sample
        #         which is illuminated at some stage
        self.sample_area     = sampleArea # the total size of the sample, of which a subset will be solved for
        self.sample_areas    = []         # sample area for each probe position 
        self.sample_area_sub = np.zeros((N,N),dtype=np.bool)       # sum of above
        self.sampleAutocs    = []         # area of the autocorrelation function for each probe position
        self.Cross           = []         # all of the cross terms with the guts cut out
        self.IllumF          = []
        self.IllumR          = []
         
        self.Sample          = np.zeros((N,N),dtype=Illum.dtype) # the current guess for the sample function

        self.iterationsILRUFT= 0

    def Adot(self, arrayX):
        """Calculate the matrix vector product [A1, A2, ...] . x  the result is an array of arrays."""
        tempAC = np.zeros((self.Nimages, self.N, self.N), dtype = self.Illum.dtype)
        for i in range(self.Nimages):
            tempAC[i] = 2.0 * np.real(self.IllumF[i] * np.conj(bg.fft2(self.IllumR[i]*arrayX))) + 0.0J
            tempAC[i] = bg.ifft2(tempAC[i]) * (-self.sampleAutocs[i])
        return tempAC

    def ATdot(self, arrayB):
        """Calculate the matrix vector product [A1T A2T ...] . [b1, b2, ...] the result is the sum of A1T . b1 + A2T . b2 ... (same dimensions as the sample function)
        
        AT means the transpose of A and a "," represents the next row."""
        arrayout = np.zeros((self.N, self.N), dtype = self.Illum.dtype)
        for i in range(self.Nimages):
            tempAC    = bg.fft2(arrayB[i])
            tempAC    = self.IllumF[i] * (2.0e0 * np.real(tempAC))
            tempAC    = np.conj(self.IllumR[i]) * bg.ifft2(tempAC)
            arrayout += tempAC
        arrayout *= (self.sample_area_sub)
        return arrayout
    
    def ilruft(self,iterations):
        """Iteratively solve the linear equations using the conjugate gradient method.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iterationsILRUFT == 0 :
            self.Sample.fill(0)
            self.d     = np.copy(self.Cross)
            self.r     = self.ATdot(self.Cross)
            self.p     = np.copy(self.r)
            self.t     = self.Adot(self.p)
            self.norm_residual  = np.sum(np.abs(self.Cross)**2)
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
            print 'residual error =', self.error_residual[-1]

        self.iterationsILRUFT += iterations
    
    def makeAreas_Illums_Cross(self):
        """Make sample-areas for each probe position in addition to the total illuminated area."""
        for i in range(self.Nimages):
            x, y            = self.posis[i]
            Illum           = bg.roll_to(self.Illum, y, x)
            self.IllumR.append(       Illum)
            self.IllumF.append(       bg.fft2(Illum))

            self.sample_areas.append( bg.overlap(self.sample_area, np.abs(Illum)))
            self.sampleAutocs.append( self.sampleAutocArea(self.sample_areas[-1])) 
            self.sample_area_sub   += self.sample_areas[-1]
            
            self.Cross.append(        bg.ifft2(self.diffs[i]) - bg.autoc(Illum))
            self.Cross[i]          *= (-self.sampleAutocs[i])

    def sampleAutocArea(self, sampleArea):
        """Calculate the domain of the sample's autocorrelation function."""
        array = bg.autoc(sampleArea + 0.0J)
        array = bg.blurthresh_mask(np.abs(array), thresh = 1.0e-8, blur = 0)
        return array

class PILRUFT_full_area(object):
    """Implement the Ptychographic Iterative Linear Retrieval Using Fourier Transforms algorithm."""

    def __init__(self, N, Illum, diffs, sampleArea, posis):
        """Initialise the PILRUFT variables."""
        self.N               = N          # the side length of the detector
        self.Illum           = Illum      # the complex illumination function
        self.diffs           = diffs      # an array of numpy arrays
        self.posis           = posis      # the list of probe positions
        self.Nimages         = len(diffs)
        #---> There are three sample areas 
        #       - The total size of the object
        #       - The overlap of the illumination with 
        #         the total sample areas for each probe 
        #         position
        #       - The complete area of the sample
        #         which is illuminated at some stage
        self.sample_area     = sampleArea # the total size of the sample, of which a subset will be solved for
        self.Cross           = []         # all of the cross terms with the guts cut out
        self.IllumF          = []
        self.IllumR          = []
         
        self.Sample          = np.zeros((N,N),dtype=Illum.dtype) # the current guess for the sample function

        self.iterationsILRUFT= 0

        for i in range(self.Nimages):
            x, y            = self.posis[i]
            Illum           = bg.roll_to(self.Illum, y, x)
            self.IllumR.append(       Illum)
            self.IllumF.append(       bg.fft2(Illum))
        
        self.Cross_calc()

    def Adot(self, arrayX):
        """Calculate the matrix vector product [A1, A2, ...] . x  the result is an array of arrays."""
        tempAC = np.zeros((self.Nimages, self.N, self.N), dtype = self.Illum.dtype)
        for i in range(self.Nimages):
            tempAC[i] = 2.0 * np.real(self.IllumF[i] * np.conj(bg.fft2(self.IllumR[i]*arrayX))) + 0.0J
            tempAC[i] = bg.ifft2(tempAC[i]) 
        return tempAC

    def ATdot(self, arrayB):
        """Calculate the matrix vector product [A1T A2T ...] . [b1, b2, ...] the result is the sum of A1T . b1 + A2T . b2 ... (same dimensions as the sample function)
        
        AT means the transpose of A and a "," represents the next row."""
        arrayout = np.zeros((self.N, self.N), dtype = self.Illum.dtype)
        for i in range(self.Nimages):
            tempAC    = bg.fft2(arrayB[i])
            tempAC    = self.IllumF[i] * (2.0e0 * np.real(tempAC))
            tempAC    = np.conj(self.IllumR[i]) * bg.ifft2(tempAC)
            arrayout += tempAC
        arrayout *= self.sample_area
        return arrayout
    
    def ilruft(self,iterations):
        """Iteratively solve the linear equations using the conjugate gradient method.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iterationsILRUFT == 0 :
            self.Sample.fill(0)
            self.d     = np.copy(self.Cross)
            self.r     = self.ATdot(self.Cross)
            self.p     = np.copy(self.r)
            self.t     = self.Adot(self.p)
            self.norm_residual  = np.sum(np.abs(self.Cross)**2)
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
            print 'residual error =', self.error_residual[-1]

        self.iterationsILRUFT += iterations
    
    def Cross_calc(self):
        """Make sample-areas for each probe position in addition to the total illuminated area."""
        for i in range(self.Nimages):
            x, y            = self.posis[i]
            Illum           = bg.roll_to(self.Illum, y, x)
            self.Cross.append(        bg.ifft2(self.diffs[i]) - bg.autoc(Illum))

    def normalise_images(self,tot=1.0):
        """Normalise the image so that sum imageAmp**2 = tot * sum |Illum|**2 ."""
        tot2           = tot * np.sum(np.real( self.Illum * np.conj(self.Illum) ))
        for i in range(self.Nimages):
            self.diffs[i]      = bg.normaliseInt(self.diffs[i], tot = tot)


class PILRUFT2(object):
    """Implement the Ptychographic Iterative Linear Retrieval Using Fourier Transforms algorithm."""

    def __init__(self,N,Illum,diffs,sampleArea,posis):
        """Initialise the PILRUFT variables."""
        self.N               = N          # the side length of the detector
        self.Illum           = Illum      # the complex illumination function
        self.diffs           = diffs      # a list of numpy arrays
        self.sample_area     = sampleArea # the total size of the sample, of which a subset will be solved for
        self.posis           = posis      # the list of probe positions
        self.probe_positions = []         # a list x,y coordinates 
        self.sample_areas    = []         # a list of numpy arrays
        self.Cross           = []         # a list of numpy arrays
        self.iterationsILRUFT= 0
        
        self.nvarvects       = []
        self.neqvects        = []
        self.xvect           = []
        self.bvects          = []
         
        self.Sample          = None       # the current guess for the sample function
        self.sample_area_sub = None
        self.neqTot          = 0

    def setVects(self):
        """Make the xvect and bvects."""
        # make all of the bvects
        #   -- get the sample area for each probe position from the overlap between the probe and the sample area
        #       -- add it to the total sample area 
        #   -- use it to calculate the sample autocorrelation area
        #   -- calculate the cross term then make that bvect
        #   -- repeat for all of the diffs
        self.sample_area_sub = np.zeros((self.N,self.N), dtype=np.float64)
        for i in range(len(self.diffs)):
            ####### deal with sample areas
            Illum                  = bg.roll_to(self.Illum, self.posis[i][1], self.posis[i][0])
            self.sample_areas[i:]  = [bg.overlap(self.sample_area,np.abs(Illum))]
            self.sample_area_sub  += self.sample_areas[i]
            
            ####### make the bvect
            Cross                  = bg.ifft2(self.diffs[i]) - bg.autoc(Illum)
            self.neqvects.append(    self.makeNeqvect(self.sample_areas[i], Cross))
            neq                    = self.neqvects[-1].shape[0]
            self.bvects[i:]        = [bg.mapping(Cross, self.neqvects[i])]
            self.Cross[i:]         = [Cross]
            self.neqTot           += neq

        self.bvect            = np.concatenate((self.bvects))
        self.sample_area_sub *= (self.sample_area_sub > 0.5)
        self.nvarvect         = self.makeNvarvect(self.sample_area_sub)
        self.nvar             = self.nvarvect.shape[0]
        self.xvect            = np.zeros((2*self.nvar),dtype=np.float64)
        
    def makeNeqvect(self, sampleArea, Cross):
        """Return neqvect (a numpy array mapping the cross terms to a 1d vector) and the number of cross terms collected.
         
        Given a sample area and an array of cross terms calculate a 1d mapping vector by selecting half of the array outside of the sample autocorrelation area."""
        #nvarvect = np.zeros((self.N**2,2),dtype = int)
        neqvect  = np.zeros((self.N**2,2),dtype = int)
        cross_area   = bg.blurthresh(np.abs(Cross), thresh=1.0e-5, blur=0)
        sample_autoc = bg.blurthresh(np.abs(bg.autoc(sampleArea)), thresh=1.0e-8, blur=0)
        neq = 0
        for i in range(self.N-1):
            for j in range(self.N/2 - 1):
                if (cross_area[i,j] > 0.5e0) and (sample_autoc[i,j] < 0.5e0) :
                    neqvect[neq,0] = i
                    neqvect[neq,1] = j
                    neq += 1
        
        j = self.N/2 - 1
        for i in range(self.N/2):
            if (cross_area[i,j] > 0.5e0) and (sample_autoc[i,j] < 0.5e0) :
                neqvect[neq,0] = i
                neqvect[neq,1] = j
                neq += 1
        
        neqvect  = np.resize(neqvect ,(neq ,2))
        
        return neqvect
    
    def makeNvarvect(self, sampleArea):
        """Return neqvect (a numpy array mapping the cross terms to a 1d vector) and the number of cross terms collected.
         
        Given a sample area and an array of cross terms calculate a 1d mapping vector by selecting half of the array outside of the sample autocorrelation area."""
        nvarvect = np.zeros((self.N**2,2),dtype = int)
        nvar = 0
        for i in range(self.N):
            for j in range(self.N):
                if self.sample_area[i,j] > 0.5e0 :
                    nvarvect[nvar,0] = i
                    nvarvect[nvar,1] = j
                    nvar += 1
        
        nvarvect = np.resize(nvarvect,(nvar,2))
        self.xvect = np.zeros((2*nvar),dtype=np.float64)
        return nvarvect
    
    def Adot(self, vect):
        pass

    def ATdot(self, vect):
        pass
    
    def ilruft(self,iterations):
        """Iteratively solve the linear equations using the conjugate gradient method.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iterationsILRUFT == 0 :
            # self.Illum_F = bg.fft2(self.Illum_R)
            self.xvect = 0.0
            self.d     = np.copy(self.bvect)
            self.r     = self.ATdot(self.bvect)
            self.p     = np.copy(self.r)
            self.t     = self.Adot(self.p)
            self.norm_residual  = np.sum(self.bvect**2)
            self.error_residual = []
        
        for i in range(iterations):
            temp        = np.sum(self.r**2)
            self.alpha  = temp / np.sum(self.t**2)
            self.xvect += self.alpha * self.p
            self.d     -= self.alpha * self.t
            self.r      = self.ATdot(self.d)
            self.betta  = np.sum(self.r**2) / temp
            self.p      = self.r + self.betta * self.p
            self.t      = self.Adot(self.p)
            #self.error_residual.append(np.sum(self.d**2)/self.norm_residual)
            self.error_residual.append(bg.l2norm(self.bvect,self.Adot(self.xvect)))
            print 'residual error =', self.error_residual[-1]
        #self.Sample     = bg.unmapping(self.xvect, self.nvarvect, self.N)
        #self.Sample     = self.ERA()
        #self.Exit       = (self.Sample + 1.0) * self.Illum_R
        self.iterationsILRUFT += iterations
