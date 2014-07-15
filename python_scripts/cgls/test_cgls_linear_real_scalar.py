from cgls import *
import os, sys, getopt, inspect
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

def f(x):
    """This is the quadratic metric to minimise.
    
    Here f(x) = 0.5 * x  . 4 . x - 40 . x + 205
              = 0.5 * xT . A . x - bT . x + c
    And so:
    A = 4
    b = 40
    c = 205
    """
    out = 2.0 * (x - 10.0)**2 + 5.0
    return out

def f_grad(x):
    """This is the derivative of f(x). When out = 0 we have the solution.
    
    Here we have: f'(x) = 4 . x - 40
    So our linear equations are A . x = b
                                4 . x = 40
    """
    out = 4.0 * x - 40.0
    return out

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
    #
    # The linear problem
    Adot  = lambda x: np.array(4.0 * x )
    bvect = np.array([40.0])
    #
    # cgls
    prob  = Cgls(Adot, bvect, x0 = np.array([0.0]))
    xs = []
    ys = []
    xs.append(0.0)
    for i in range(len(bvect)):
        prob.cgls(iterations = 1)
        xs.append(prob.x[-1])
    for x in xs:
        ys.append(f(x))
    #
    # Steepest descent
    prob2 = Steepest(Adot, bvect, x0 = np.array([15.0]))
    xs2 = []
    ys2 = []
    xs2.append(15.0)
    for i in range(len(bvect)):
        prob2.sd(iterations = 1)
        xs2.append(prob2.x[-1])
    for x in xs2:
        ys2.append(f(x))
    #
    # plot f and the path that cgls took
    plt.clf()
    gs = GridSpec(1,1)
    gs.update(hspace=0.5)
    ax = plt.subplot(gs[0,0])
    # 
    # plot f
    x2 = np.linspace(0.0, 20.0, 1000)
    ax.plot(x2, f(x2), linewidth=2, alpha=0.5, label='f(x)')
    # 
    # plot the path
    ax.plot(xs, ys, linewidth=2, alpha=0.5, label='cgls')
    ax.set_title('metric', fontsize=18, position=(0.5, 1.01))
    # 
    # plot the path
    ax.plot(xs2, ys2, linewidth=2, alpha=0.5, label='steepest')
    #
    plt.legend()
    plt.gcf().set_size_inches(10,5)
    #
    plt.savefig(outputDir + 'test_cgls_linear_real_scalar.png')




