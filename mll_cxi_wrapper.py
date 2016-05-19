"""
Load an mll cxi file with:
process_2/dark
process_2/whitefield
process_2/background
process_2/ptycho_mask
"""

import numpy as np
import time
import sys
import h5py
import scipy.constants as sc


from Ptychography import write_cxi
from Ptychography import DM
from Ptychography import ERA
from Ptychography import utils

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



if __name__ == '__main__':
	params = utils.parse_cmdline_args()

	if rank == 0 :
		f = h5py.File(params['input']['cxi_fnam'], 'r')
		 
		# get the sample positions
		fast_axis = f['entry_1/instrument_1/motor_positions/scan_axes/Fast axis/name'].value
		slow_axis = f['entry_1/instrument_1/motor_positions/scan_axes/Slow axis/name'].value
		mll1_name   = f['entry_1/sample_1/name'].value
		mll2_name   = f['entry_1/sample_2/name'].value
		sample_name = f['entry_1/sample_3/name'].value

		# get the beam energy 
		E = f['entry_1/instrument_1/source_1/energy'].value 

		# get the fast scan values
		fast_values = f['entry_1/instrument_1/motor_positions/scan_axes/Fast axis/POSITION'].value
		slow_values = f['entry_1/instrument_1/motor_positions/scan_axes/Slow axis/POSITION'].value
		steps = [f['entry_1/instrument_1/motor_positions/scan_axes/Slow axis/Steps'].value, 
				 f['entry_1/instrument_1/motor_positions/scan_axes/Fast axis/Steps'].value]
				

		print '\nThis is a slow scan of:', slow_axis
		print 'From:', slow_values[0], 'to:', slow_values[-1], 'steps:', steps[0]
		print '\nThis is a fast scan of:', fast_axis
		print 'From:', fast_values[0], 'to:', fast_values[-1], 'steps:', steps[1]

		print '\nMLL1 name              :', mll1_name
		print 'MLL2 name              :', mll2_name
		print 'Sample name            :', sample_name
		print 'Beam energy (keV)      : {0:.2f}'.format( 1.0e-3 * E / sc.e )

		# get the data
		print '\nReading data...'
		if params['input']['frames'] == -1 or params['input']['frames'] == None :
			frames = f['entry_1/data_1/data'].shape[0]
		else :
			frames = params['input']['frames'] 
		
		if params['input']['every_n_frames'] == -1 or params['input']['every_n_frames'] == None :
			every = 1
		else :
			every = params['input']['every_n_frames'] 
		I     = []
		steps = []
		for step in range(0, f['entry_1/data_1/data'].shape[0], every):
			I.append(f['entry_1/data_1/data'][step])
			steps.append(step)
			if len(I) == frames:
				break
		I = np.array(I)
		print '\nprocessing frames:',I.shape, I.dtype
		print '\nprocessing steps:',steps


	#
	# mask
    
	#
	# Interferometer

	#
	# whitefield

	# background

	# normalisation
	#def ERA(I, R, P, O, iters, OP_iters = 1, mask = 1, background = None, method = None, Pmod_probe = False, probe_centering = False, hardware = 'cpu', alpha = 1.0e-10, dtype=None, full_output = True):
