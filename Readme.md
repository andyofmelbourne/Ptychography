# Ptychography
ptychography toolbox module for python

### To install as the local user on Linux
```
$ cd ~/.local/lib/python2.7/site-packages/

# If this fails then 
$ mkdir -p ~/.local/lib/python2.7/site-packages/
$ cd ~/.local/lib/python2.7/site-packages/

$ git clone https://github.com/andyofmelbourne/Ptychography.git
```
Done!

### Examples
#### ERA on a single cpu
This will perform reconstructions updating just the object, the probe 
or both with and without background correction using the error reduction
algorithm.
```
$ python ~/.local/lib/python2.7/site-packages/Ptychography/ptychography/era.py
```

Now you should have six files in your local directory "output_method[N].cxi".
To have a look at the retrievals run:
```
$ python ~/.local/lib/python2.7/site-packages/Ptychography/ptychography/display.py output_method1.cxi
```

#### ERA on 2 cpu cores
To test the mpi routines run:
```
$ mpirun -n 2 python ~/.local/lib/python2.7/site-packages/Ptychography/ptychography/era.py
```

### To run on the gpu machines
```
$ module load python/2.7
$ module load opencl/intel
$ export PYTHONPATH=/afs/desy.de/user/a/amorgan/python_packages/lib/python2.7/site-packages/:$PYTHONPATH
$ export PYTHONPATH=/nfs/cfel/cxi/home/amorgan/2015/PETRA-P11-Oct/Ptychography/:$PYTHONPATH
```

### To run on the max-cfel001 or max-cfel002 machines
I had to install mpi4py so this will only work for me (Andrew) at this stage:
```
$ ssh -X max-cfel001
$ export PYTHONPATH=/nfs/cfel/cxi/home/amorgan/2015/PETRA-P11-Oct/Ptychography/:$PYTHONPATH
$ cd PETRA-P11-Oct/scan_222
$ /usr/lib64/openmpi/bin/mpirun -np 32 python test.py
```
