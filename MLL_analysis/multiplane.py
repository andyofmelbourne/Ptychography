# Run the sequence for a number of planes simultaneously

# Just run pipeline many times

import sys
import os, sys, getopt
import subprocess 

# Parameters
##########################################
##########################################
tempdata_dir = '../../../tempdata/MLL_calc_multi/'
tempdata_dirs = []
tempdata_dirs.append('../../../tempdata/MLL_calc_multi/run0/')
tempdata_dirs.append('../../../tempdata/MLL_calc_multi/run1/')
tempdata_dirs.append('../../../tempdata/MLL_calc_multi/run2/')
tempdata_dirs.append('../../../tempdata/MLL_calc_multi/run3/')
tempdata_dirs.append('../../../tempdata/MLL_calc_multi/run4/')
##########################################
##########################################


for i, t in enumerate(tempdata_dirs):
    subprocess.Popen([sys.executable,"pipeline.py","--location=cfelsgi","--run="+str(i), "--tempdata_dir="+t])


