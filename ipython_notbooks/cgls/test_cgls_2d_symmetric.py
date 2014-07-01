from cgls import *
import os, sys, getopt, inspect
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from pylab import cm

def myShow(X, Y, Z, xs = None, xs2 = None, xlabel = None, xlabel2=None, xdot=None, ydot=None, title = None):
    gs = GridSpec(1, 1)
    gs.update(top = 0.95, bottom = 0.02,left=0.02, right=0.98, hspace=0.2,wspace=0.05)
    ax = plt.subplot(gs[0, 0])
    m = 300
    V = np.array(range(50))**2 * np.sqrt(Z.max() - Z.min()) / np.float(m - 1) + Z.min()
    
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
    fig.set_size_inches(10, 10)

def f(x):
    """Here:

    f(x) = 0.5 * xT . A . x - bT . x + c
    A    = [[3, 2], [2, 6]]
    b    = [2, -8]
    c    = 2
    """
    b  = np.array([2.0, -8.0])
    f  = Adot(x)
    f  = 0.5 * np.sum(x * f) - np.sum(b * x) + 2
    return f
    
def Adot(x):
    A = np.array([[3.0, 2.0], [2.0, 6.0]])
    return np.dot(A, x)

def main(argv):
    inputdir = './'
    outputdir = './'
    try :
        opts, args = getopt.getopt(argv,"hi:o:",["inputdir=","outputdir="])
    except getopt.GetoptError:
        print 'test_cgls_linear_real_scalar.py -i <inputdir> -o <outputdir>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test_cgls_linear_real_scalar.py -i <inputdir> -o <outputdir>'
            sys.exit()
        elif opt in ("-i", "--inputdir"):
            inputdir = arg
        elif opt in ("-o", "--outputdir"):
            outputdir = arg
    return inputdir, outputdir

if __name__ == '__main__':
    inputDir, outputDir = main(sys.argv[1:])
    #--------------------------
    # Steepest descent
    #--------------------------
    x0 = np.array([-2.0, -2.0])
    sd = Steepest(Adot, np.array([2.0, -8.0]), x0=x0)
    #
    xs = []
    xs.append(x0)
    for i in range(40):
        x = sd.sd(1)
        xs.append(x)
    #--------------------------
    # cgls
    #--------------------------
    x0 = np.array([-2.0, -2.0])
    cgls = Cgls(Adot, np.array([2.0, -8.0]), x0=x0)
    #
    xs_cgls = []
    xs_cgls.append(x0)
    for i in range(2):
        x = cgls.cgls(1)
        xs_cgls.append(x)
    #--------------------------
    # plot
    #--------------------------
    N = 100
    X, Y = np.meshgrid(np.linspace(-4.0, 6.0, N), np.linspace(-6.0, 4.0, N))
    Z = []
    for x, y in zip(X.flatten(), Y.flatten()):
        Z.append(f(np.array([x,y])))
    Z = np.array(Z).reshape(X.shape)
    #
    myShow(X, Y, Z, xs = xs, xs2=xs_cgls, xdot=2.0, ydot=-2.0, title = 'f(x)', xlabel='steepest descent', xlabel2='cgls')
    #
    plt.savefig(outputDir + 'test_cgls_linear_real_2d.png')




