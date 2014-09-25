from andrew import STEM as st
from andrew import bagOfns as bg
from andrew import phaseRetrieval as ph
import numpy as np

def makePupilPhase(aperture, flip=1):
    """Makes the STEM probe pupil phase from the aberration coefficients in the paper

    paper = Deterministic electron ptychography at atomic resolution
            PHYSICAL REVIEW B 89, 064101 (2014)
            A. J. D`Alfonso, A. J. Morgan, A. W. C. Yan, P. Wang, H. Sawada, A. I. Kirkland, and L. J. Allen
    """
    # Make the STEM probe #################################
    probe = st.STEMprobe()
    
    # work out the geometry ###############################
    probe.N      = 1024
    probe.energy = 300.0e3
    probe.ampF   = aperture
    probe.drad   = probe.dradFromAperture(aperture, 24.0e-3)
    probe.makeParams()
    probe.disParams()

    probe.aberrations['C1'] = 91.0e1           * np.exp(2.0J * np.pi * 0.0  )
    probe.aberrations['A1'] = 20.7             * np.exp(2.0J * np.pi *-0.07 )
    probe.aberrations['B2'] = 23.0e1           * np.exp(2.0J * np.pi *-0.95 )
    probe.aberrations['A2'] = 23.4e1           * np.exp(2.0J * np.pi * 0.09 )
    probe.aberrations['C3'] = 27.8e3           * np.exp(2.0J * np.pi * 0.0  )
    probe.aberrations['S3'] = 43.5e1           * np.exp(2.0J * np.pi *-0.07 )
    probe.aberrations['A3'] = 19.1e2           * np.exp(2.0J * np.pi *-0.02 )
    probe.aberrations['B4'] = 55.0e2           * np.exp(2.0J * np.pi * 0.44 )
    probe.aberrations['D4'] = 37.0e2           * np.exp(2.0J * np.pi *-0.13 )
    probe.aberrations['A4'] = 10.6e4           * np.exp(2.0J * np.pi * 0.10 )
    probe.aberrations['C5'] =-70.4e5           * np.exp(2.0J * np.pi * 0.00 )
    probe.aberrations['A5'] = 48.2e5           * np.exp(2.0J * np.pi * 0.04 )    

    probe.makePhase()
    probe.phaseF = bg.orientate(probe.phaseF,flip)
    return probe
