# Ptychography
ptychography toolbox module for python

### To install as the local user on Linux
```
$ git clone https://github.com/andyofmelbourne/Ptychography.git ~/.local/lib/python2.7/site-packages/Ptychography
```
Done!

### Requires
- python (probably >= 2.7)
- scipy
- numpy

And for display and testing routines:
- h5py 
- pyqtgraph

### Examples

#### command line usage
This will perform reconstructions updating just the object, the probe 
or both with and without background correction using the difference map, 
and the error reduction algorithm.
```
$ cp ~/.local/lib/python2.7/site-packages/Ptychography/test.py .
$ python test.py
```
##### or with 2 cpu cores
```
$ mpirun -n 2 python test.py
```

Now you should have a file your local directory "output_method1.cxi".
To have a look at the retrievals run:
```
$ python ~/.local/lib/python2.7/site-packages/Ptychography/ptychography/display.py output_method1.cxi
```

#### ERA on a single cpu, within python
```
>>> import Ptychography as pty
>>> I, R, M, P, O, B = pty.forward_sim()
>>> O_ret, info      = pty.ERA(I, R, P, None, iters=100, mask=M, method=1)
```

### To run on the max-cfel machines
```
$ ssh -X max-cfel
$ source /nfs/cfel/cxi/common/cfelsoft-rh7/anaconda-py2/anaconda-setup-bash.sh 
$ cp ~/.local/lib/python2.7/site-packages/Ptychography/test.py .
$ mpirun -n 2 python test.py
```
