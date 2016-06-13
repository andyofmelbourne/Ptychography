import os, sys, getopt, inspect
import numpy as np

import cgls_nonlinear as cg

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from pylab import cm

def myShow(X, Y, Z, xs = None, xs2 = None, xlabel = None, xlabel2=None, xdot=None, ydot=None, title = None):
    gs = GridSpec(1, 1)
    gs.update(top = 0.95, bottom = 0.02,left=0.02, right=0.98, hspace=0.2,wspace=0.05)
    ax = plt.subplot(gs[0, 0])
    m = 30
    V = np.array(range(10))**2 * np.sqrt(Z.max() - Z.min()) / np.float(m - 1) + Z.min()
    
    ax.contourf(X, Y, Z, V, alpha=.75, cmap=cm.RdBu)
    C = ax.contour(X, Y, Z, V, colors='black', linewidth=.3, alpha=0.5)
    # put the axes in the centre
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    # put white boxes around the labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(10)
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.35 ))
    if xdot != None and ydot != None :
        ax.plot([xdot,xdot],[ydot,ydot], marker='o', color='black')
    if xs != None:
        x = []
        y = []
        for xx,yy in xs:
            x.append(xx)
            y.append(yy)    
        ax.plot(x, y, label=xlabel)
    if xs2 != None:
        x = []
        y = []
        for xx,yy in xs2:
            x.append(xx)
            y.append(yy)    
        ax.plot(x, y, label=xlabel2)
    if xlabel != None:
        ax.legend(loc='upper left')
    if title !=None:
        ax.set_title(title, fontsize=18, position=(0.5, 1.01))
    fig = plt.gcf()
    fig.set_size_inches(10, 5)


a = [0.3, 1.0]
b = 0.5
c = -40.0
x0 = 8.0
def f(x):
    f = (a[0]*x[0]**2 + a[1]*x[1]**2) + c * np.exp( -b * ((x[1] - x0)**2 + x[0]**2))
    return f

def df(x):
    out_y = 2 * a[0] * x[0] - 2 * b * x[0] * c * np.exp( -b * ((x[1] - x0)**2 + x[0]**2))
    out_x = 2 * a[1] * x[1] - 2 * b * (x[1]-x0) * c * np.exp( -b * ((x[1] - x0)**2 + x[0]**2))
    return np.array([out_y, out_x])

def fd(x, d):
    return np.dot(d, df(x))

def dfd(x, d):
    dsum = np.sum(np.abs(d)**2)
    out  = 2 * (a[0]*d[0]**2 + a[1]*d[1]**2) + 2 * b * ( 2 * b * (d[0]*x[0] + d[1] * (x[1] - x0))**2 - dsum) * c * np.exp( -b * ((x[1] - x0)**2 + x[0]**2))
    return out

shape = (200, 200)
X, Y = np.meshgrid(np.linspace(-20, 20, shape[1]), np.linspace(-10, 10, shape[1]))

F = []
for x, y in zip(X.flatten(), Y.flatten()):
    F.append(f(np.array([x,y])))

F = np.array(F).reshape(X.shape)


xs = []
xs.append([-0.8, 8])

sd = cg.Steepest(xs[-1], f, df, fd, dfd)
#
for i in range(10):
    xs.append(sd.sd(iterations=1))


xs2 = []
xs2.append([-0.8, 8])

cgls = cg.Cgls(xs2[-1], f, df, fd, dfd)
#
for i in range(10):
    xs2.append(cgls.cgls(iterations=1))
print xs2

myShow(X, Y, F, xs = xs , xs2=xs2, title = 'f(x)', xlabel='steepest descent', xlabel2='cgls')

plt.savefig('test_cgls_nonlinear_2d_real.png')




