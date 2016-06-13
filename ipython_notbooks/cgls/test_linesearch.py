import numpy as np
import line_search as ls
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt


a = 1.0
b = 1.0
c = 40.0
x0 = 0.0
def f(x):
    f = a * x**2 + c * np.exp( -b * (x - x0)**2)
    return f

def fd(x, d):
    out = d * (2 * a * x - 2 * b * (x - x0) * c * np.exp( -b * (x - x0)**2))
    return out

def dfd(x, d):
    out = 2 * a - 2 * c * b * np.exp( -b * (x - x0)**2) + 4 * (x - x0)**2 * b**2 * c * np.exp( -b * (x - x0)**2)
    out = d * out * d
    return out

xs = []
ys = []
xs.append(0.9)
ys.append(f(xs[-1]))
for i in range(1000):
    xx, status = ls.line_search_newton_raphson(xs[-1], 1, fd, dfd, iters = 1, tol=1.0e-10)
    if status == False :
        print xx, status
    xs.append(xx)
    ys.append(f(xs[-1]))

xs2 = []
ys2 = []
xs2.append(-0.9)
ys2.append(f(xs2[-1]))
for i in range(1000):
    xx, status = ls.line_search_secant(xs2[-1], 1, fd, iters = 1, tol=1.0e-10)
    if status == False :
        print xx, status
    xs2.append(xx)
    ys2.append(f(xs2[-1]))


x = np.linspace(-10, 10, 1000)

gs = GridSpec(1, 1)
ax = plt.subplot(gs[0, 0])
ax.plot(x, f(x), label='f(x)', linewidth=3, alpha = 0.4)
ax.plot(xs, ys, 'k', label='newton')
ax.plot(xs2, ys2, 'g', label='secant')
ax.legend()

fig = plt.gcf()
fig.set_size_inches(15, 6)

plt.savefig('test_linesearch.png')
