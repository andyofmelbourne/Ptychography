# Algorithm Notes

Mod sqaure of a complex array: 
    use ( a.conj() * a ).real
```
$ a = np.random.random((100, 512, 512)) + 1J*np.random.random((100, 512, 512))

$ %timeit ( a.conj() * a ).real
1 loops, best of 3: 309 ms per loop

$ %timeit np.abs(a)**2
1 loops, best of 3: 956 ms per loop
```

Rolling an array:
    use multiroll(a, [0, 5, 10])
```
$ %timeit -n10 np.roll(np.roll(a, 5, 1), 10, 2)
10 loops, best of 3: 645 ms per loop

$ %timeit -n10 pt.era.multiroll(a, [0, 5, 10])
10 loops, best of 3: 200 ms per loop

$ np.allclose(np.roll(np.roll(a, 5, 1), 10, 2), pt.era.multiroll(a, [0,5,10]))
True
```

MPI native vs pickle allreduce:
    use native
```
$ mpirun -n 4 python ptychography/test_MPI.py
numpy:  delta t: 2.09517788887
pickle: delta t: 16.0462508202
```
about 8 x faster


MPI it-hpc-cxi01 vs max-cfel001 :
    use max-cfel001
```
$ mpirun -np 12 python ptychography/era.py mpi 100
it-hpc-cxi01, delta t: 3.27306580544
max-cfel001 , delta t: 0.4412150383
```
7.4 x speedup (updating obj, probe and background)

probe_centering = True really helps when updating both O and P
