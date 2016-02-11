# Ptychography
ptychography toolbox module for python

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
