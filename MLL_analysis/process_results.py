# Process the results and spit out some figures for easy viewing
#   Maybe this can also be pasted into an Ipython notebook for further analysis...

# 

import os, sys, getopt, inspect
import numpy as np
import scipy as sp
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import process_diffs as pd
import colorsys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from python_scripts import bagOfns as bg



def main(argv):
    inputdir = './'
    outputdir = './'
    scan = '0181'
    run = 0
    try :
        opts, args = getopt.getopt(argv,"hi:o:sr",["inputdir=","outputdir=","scan=","run="])
    except getopt.GetoptError:
        print 'process_diffs.py -i <inputdir> -o <outputdir>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'process_diffs.py -i <inputdir> -o <outputdir> -s <scan> -r <run>'
            sys.exit()
        elif opt in ("-i", "--inputdir"):
            inputdir = arg
        elif opt in ("-o", "--outputdir"):
            outputdir = arg
        elif opt in ("-s", "--scan"):
            scan = arg
        elif opt in ("-r", "--run"):
            run = int(arg)
    return inputdir, outputdir, scan, run

def hls_to_rgb2(hs, ls, ss):
    h2 = hs.flatten()
    l2 = ls.flatten()
    s2 = ss.flatten()
    array_rgb = []
    for h, l, s in zip(h2, l2, s2):
        array_rgb.append(colorsys.hls_to_rgb(h, l, s))
    array_rgb = np.array(array_rgb)
    return array_rgb.reshape((hs.shape[0], hs.shape[1], 3))

def hvs_to_rgb2(hs, vs, ss):
    h2 = hs.flatten()
    v2 = vs.flatten()
    s2 = ss.flatten()
    array_rgb = []
    for h, v, s in zip(h2, v2, s2):
        array_rgb.append(colorsys.hsv_to_rgb(h, s, v))
    array_rgb = np.array(array_rgb)
    return array_rgb.reshape((hs.shape[0], hs.shape[1], 3))

def complex_to_rgb(complex_data, invert=False, saturation = 1.0, rot = np.exp(1J * 0 * np.pi), scale=[0.0,1.0]):
    phase = np.angle(complex_data * rot)
    amplitude = np.abs(complex_data)
    l = bg.scale(amplitude, scale[0], scale[1])
    s = np.ones_like(l) * saturation
    h = (phase + np.pi) / (2.0 * np.pi)
    rgb = hvs_to_rgb2(h, l, s)
    if(invert):
        return 1-rgb
    else:
        return rgb
    
def colourwheel(pos = [0.35, 0.15, 0.2, 0.2]):
    N = 512
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X,Y = np.meshgrid(x,y)
    R = np.sqrt(X*X + Y*Y)
    PHI = np.arctan2(Y, X)
    ax = plt.axes(pos, polar=True)
    ax.imshow(complex_to_rgb(R*np.exp(1j*PHI)  * (R<1)))
    ax.set_xticks([-0.5, 0, np.pi/2, np.pi, 3*np.pi/2])#-.5)
    ax.set_yticks([0, N/3, 2*N/3, N])
    ax.set_xticklabels([])#'', '$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$'])
    ax.set_yticklabels([])
    return ax

def fnamBase_match(fnam):
    fnam_base  = os.path.basename(fnam)
    fnam_dir   = os.path.abspath(os.path.dirname(fnam))
    onlyfiles  = [ f for f in os.listdir(fnam_dir) if os.path.isfile(os.path.join(fnam_dir,f)) ]
    fnam_match = [ f for f in onlyfiles if f[:len(fnam_base)] == fnam_base ]
    try : 
        fnam_match[0]
    except :
        return False
    return True

def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def makeDiff(sample, probe, coord):
    return np.abs(bg.fft2(makeExit(sample, probe, coord)))**2

def makeExit(sample, probe, coord):
    exit = bg.roll(sample, coord)[:probe.shape[0], :probe.shape[1]]
    exit *= probe 
    return exit

def makeExits(sample, probe, coords):
    exits = np.zeros((len(coords), probe.shape[0], probe.shape[1]), dtype=np.complex128)
    for i, coord in enumerate(coords):
        exits[i] = bg.roll(sample, coord)[:probe.shape[0], :probe.shape[1]]
    exits *= probe 
    return exits

def makeHeatmap(probe, sample, ij_coords):
    temp       = np.zeros_like(sample, dtype=np.float64)
    heatmap    = np.zeros_like(sample, dtype=np.float64)
    for i in range(len(ij_coords)):
        temp[:probe.shape[0], :probe.shape[1]] = np.abs(probe)**2
        heatmap += bg.roll(temp, [-ij_coords[i][0], -ij_coords[i][1]])
    return heatmap

def make_sample_fig(sample_init, sample_ret, sample_support, heatmap, outputDir):
    plt.clf()
    gs = GridSpec(5,2)
    gs.update(hspace=0.5)
    #
    ax = plt.subplot(gs[0,0])
    ax.imshow(sample_support, aspect='auto')
    ax.set_title('sample support', fontsize=18, position=(0.5, 1.01))
    ax = plt.subplot(gs[0,1])
    ax.imshow(heatmap, aspect='auto', cmap='Greys_r', interpolation='nearest')
    ax.set_title('heatmap', fontsize=18, position=(0.5, 1.01))
    #
    ax = plt.subplot(gs[1,0])
    ax.imshow(np.abs(sample_ret), aspect='auto')#, vmin = np.abs(sample_init).min(), vmax = np.abs(sampleInit).max())
    ax.set_title('sample ret amp', fontsize=18, position=(0.5, 1.01))
    ax.set_ylim([0.7, 1.0])
    ax = plt.subplot(gs[2,0])
    ax.imshow(np.abs(sample_init), aspect='auto')#, vmin = np.abs(sample_init).min(), vmax = np.abs(sample_init).max())
    ax.set_title('sample init amp', fontsize=18, position=(0.5, 1.01))
    #
    ax = plt.subplot(gs[1,1])
    ax.imshow(np.angle(sample_ret), aspect='auto')#, vmin = np.angle(sample_init).min(), vmax = np.angle(sample_init).max())
    ax.set_title('sample ret phase', fontsize=18, position=(0.5, 1.01))
    ax = plt.subplot(gs[2,1])
    ax.imshow(np.angle(sample_init), aspect='auto')#, vmin = np.angle(sample_init).min(), vmax = np.angle(sample_init).max())
    ax.set_title('sample init phase', fontsize=18, position=(0.5, 1.01))
    #
    ax = plt.subplot(gs[3,0])
    ax.plot(np.sum(np.abs(sample_ret), axis=0)/float(sample_ret.shape[0]), linewidth=2, alpha=0.5)
    ax.set_title('sample ret amp', fontsize=18, position=(0.5, 1.01))
    #ax.set_ylim([0.90, 1.02])
    ax.set_xlim([0, sample_init.shape[1]])
    #
    ax = plt.subplot(gs[4,0])
    ax.plot(np.sum(np.abs(sample_init), axis=0)/float(sample_init.shape[0]), linewidth=2, alpha=0.5)
    ax.set_title('sample init amp', fontsize=18, position=(0.5, 1.01))
    ax.set_xlim([0, sample_init.shape[1]])
    #
    ax = plt.subplot(gs[3,1])
    ax.plot(np.sum(np.angle(sample_ret), axis=0)/float(sample_ret.shape[0]), linewidth=2, alpha=0.5)
    ax.set_title('sample ret phase', fontsize=18, position=(0.5, 1.01))
    #ax.set_ylim([-0.5, 2.0])
    ax.set_xlim([0, sample_init.shape[1]])
    #
    ax = plt.subplot(gs[4,1])
    ax.plot(np.sum(np.angle(sample_init), axis=0)/float(sample_init.shape[0]), linewidth=2, alpha=0.5)
    ax.set_title('sample init phase', fontsize=18, position=(0.5, 1.01))
    ax.set_xlim([0, sample_init.shape[1]])
    #
    plt.gcf().set_size_inches(20,15)
    #
    plt.savefig(outputDir + 'fig_sampleInit_Vs_sample_ret.png')

def make_probe_fig(probe_init, probe_ret, outputDir, scan = '0181', run = 0):
    plt.clf()
    #------------------------------------
    # Loading probe stuff
    #------------------------------------
    #
    print 'loading metadata...'
    zyxN_stack, fnams_stack = pd.load_metadata(scan = scan)
    #
    # Calculate geometry
    print 'calculating the geometry...'
    X, lamb = pd.geometry()

    spacing = [X / probe_init.shape[0], X / probe_init.shape[1]]


    fontsize = 16

    gs = GridSpec(6, 2)
    gs.update(wspace=0.1, hspace=0.3)

    #---------------------------------------
    # 2d profile focus retrieved
    #---------------------------------------
    ax = plt.subplot(gs[0,0])
    dy = 10
    dx = 40
    probe_0 = pd.probe_z(probe_ret, [- zyxN_stack[run][0][0]], spacing, lamb)
    #
    # take a slice
    probe_0 = probe_0[probe_0.shape[0]/2 - 1 - dy: probe_0.shape[0]/2 - 1 + dy, probe_0.shape[1]/2 - 1 - dx: probe_0.shape[1]/2 - 1 + dx]
    #
    rgb = complex_to_rgb(probe_0.T)
    #
    ax.imshow(rgb, extent=(0, 1.0e9*X*dx/float(probe_ret.shape[1]), 0, 1.0e9*X*dy/float(probe_ret.shape[1])))#, interpolation='nearest')
    ax.set_xticks([])
    ax.set_ylabel(r'nm', fontsize=fontsize)
    ax.set_title('probe amplitude focus profile to scale: run =' + str(run), fontsize=fontsize, position=(0.5, 1.01))
    #---------------------------------------
    # 2d profile focus for initial
    #---------------------------------------
    ax = plt.subplot(gs[1,0])
    probe_0 = pd.probe_z(probe_init, [- zyxN_stack[run][0][0]], spacing, lamb)
    #
    # take a slice
    probe_0 = probe_0[probe_init.shape[0]/2 - 1 - dy: probe_init.shape[0]/2 - 1 + dy, probe_init.shape[1]/2 - 1 - dx: probe_init.shape[1]/2 - 1 + dx]
    #
    rgb = complex_to_rgb(probe_0.T)
    #
    ax.imshow(rgb, extent=(0, 1.0e9*X*dx/float(probe_init.shape[1]), 0, 1.0e9*X*dy/float(probe_init.shape[1])))#, interpolation='nearest')
    ax.set_xlabel(r'nm', fontsize=fontsize)
    ax.set_ylabel(r'nm', fontsize=fontsize)
    ax.set_title('probe amplitude focus profile to scale: input', fontsize=fontsize, position=(0.5, 1.01))

    #---------------------------------------
    # line profiles
    #---------------------------------------
    ax = plt.subplot(gs[0:2, 1])
    dx = 50
    # propagate to focus
    probe_0 = pd.probe_z(probe_ret, [- zyxN_stack[run][0][0]], spacing, lamb)
    #
    # take a line
    probe_0 = probe_0[probe_ret.shape[0]/2 - 1, probe_ret.shape[1]/2 - 1 - dx: probe_ret.shape[1]/2 - 1 + dx]
    #
    x = np.linspace(- X / float(probe_init.shape[1]) * dx, X / float(probe_init.shape[1]) * dx, probe_0.shape[0]) * 1.0e9
    ax.plot(x, (np.abs(probe_0)), label = 'run '+str(run), linewidth=2, alpha = 0.5)
    ax.set_xlabel('nm', fontsize = fontsize)
    ax.set_ylabel('amplitude', fontsize = fontsize)
        
    # propagate to focus
    probe_0 = pd.probe_z(probe_init, [- zyxN_stack[run][0][0]], spacing, lamb)
    #
    # take a line
    probe_0 = probe_0[probe_init.shape[0]/2 - 1, probe_init.shape[1]/2 - 1 - dx: probe_init.shape[1]/2 - 1 + dx]
    #
    ax.plot(x, (np.abs(probe_0)), label = 'input')
    ax.legend()

    #---------------------------------------
    # In plane
    #---------------------------------------
    ax = plt.subplot(gs[2:4, 0:2])
    #
    ax = plt.subplot(gs[2,0])
    ax.imshow(np.abs(probe_ret), aspect='auto')
    ax.set_title('probe ret amp sample-plane', fontsize=fontsize, position=(0.5, 1.01))
    #
    ax = plt.subplot(gs[3,0])
    ax.imshow(np.abs(probe_init), aspect='auto')
    ax.set_title('probe init amp sample-plane', fontsize=fontsize, position=(0.5, 1.01))
    #
    ax = plt.subplot(gs[2,1])
    ax.imshow(np.log(1.0e-25+np.abs(bg.fft2(probe_ret))**2), aspect='auto')
    ax.set_title('probe ret intensity detector (log scale)', fontsize=fontsize, position=(0.5, 1.01))
    #
    ax = plt.subplot(gs[3,1])
    ax.imshow(np.log(1.0e-25+np.abs(bg.fft2(probe_init))**2), aspect='auto')
    ax.set_title('probe init intensity detector (log scale)', fontsize=fontsize, position=(0.5, 1.01))

    #---------------------------------------
    # farfield phase
    #---------------------------------------
    ax      = plt.subplot(gs[4:5, 0])
    qx      = np.linspace(-probe_init.shape[1] / 2 + 1, probe_init.shape[1] / 2, probe_init.shape[1]) / X
    mrads   = np.arctan( lamb * qx ) * 1.0e3
    # 
    # farfield phase and intensity
    probe_0     = pd.probe_z(probe_ret, [- zyxN_stack[run][0][0]], spacing, lamb)
    probeInit_0 = pd.probe_z(probe_init, [- zyxN_stack[run][0][0]], spacing, lamb)
    phase     = np.angle(bg.fft2(probe_0))
    phaseInit = np.angle(bg.fft2(probeInit_0))
    intensity = np.abs(bg.fft2(probe_0))**2
    #
    # Find the most intense pixels
    a = np.argmax(intensity, axis = 0)
    #
    # get their phase
    line_phase = []
    line_phaseInit = []
    for x,y in enumerate(a):
        line_phase.append(phase[y, x])
        line_phaseInit.append(phaseInit[y, x])
    #
    # don't look at pixels with low intensity
    line_phase     = np.array(line_phase) * (np.sum(intensity, axis=0) / np.max(np.sum(intensity, axis=0)) > 0.2)
    line_phaseInit = np.array(line_phaseInit) * (np.sum(intensity, axis=0) / np.max(np.sum(intensity, axis=0)) > 0.2)
    #
    # set the range
    range_q = np.arange(int(probe_init.shape[1]/2 *0.25), int(probe_init.shape[1]/2), 1)
    #
    # plot
    ax.plot(mrads[range_q], np.array(line_phase)[range_q], label='run '+str(run), linewidth=2, alpha = 0.5, color='black')
    ax.plot(mrads[range_q], np.array(line_phaseInit)[range_q], label='Input', linewidth=2, alpha = 0.5)
    ax.set_ylabel('radians', fontsize = fontsize)
    ax.set_xlabel(r'radians $\times 10^3$', fontsize = fontsize)
    ax.set_xlim([mrads[range_q][0], mrads[range_q][-1]])
    ax.legend()

    #---------------------------------------
    # farfield intensity
    #---------------------------------------
    ax = plt.subplot(gs[4:5, 1])
    ax.plot(mrads[range_q], 1.0e-4*np.sum(intensity, axis=0)[range_q], label='run '+str(run), linewidth=2, alpha = 0.5, color='black')
    ax.plot(mrads[range_q], 1.0e-4*np.sum(np.abs(bg.fft2(probeInit_0))**2, axis=0)[range_q], label='input', linewidth=2, alpha = 0.5)
    ax.set_ylabel(r'total counts ($\times 10^{-4}$)', fontsize = fontsize)
    ax.set_xlabel(r'radians $\times 10^{3}$', fontsize = fontsize)
    ax.legend()
    ax.set_xlim([mrads[range_q][0], mrads[range_q][-1]])

    # colour wheel 
    ax = colourwheel(pos = [0.45, 0.847, 0.04, 0.04])

    fig = plt.gcf()
    fig.set_size_inches(20,20)
    #
    plt.savefig(outputDir + 'fig_probeInit_Vs_probe_ret.png')

def make_error_fig(ij_coords, mask, sample_init, sample_ret, sample_support, probe_init, probe_ret, eMod, eSup, inputDir, outputDir):
    #
    print 'Loading the diffraction data...'
    diffs = bg.binary_in(inputDir + 'diffs', dt=np.float64, dimFnam=True)
    diffs = list(diffs)
    #
    print 'Calculating the errors...'
    lines_diff = []
    lines_ret  = []
    lines_init = []
    e_ret     = []
    e_init    = []
    diffs_sum = []
    for i in range(len(diffs)):
        update_progress(i / max(1.0, float(len(diffs)-1)))
        diff       = np.array(diffs[i], dtype=np.float64)
        diff       = mask * diff
        diff_ret   = mask * makeDiff(sample_ret, probe_ret, ij_coords[i])
        diff_init  = mask * makeDiff(sample_init, probe_init, ij_coords[i])
        diffs_sum.append(np.sum(diff))
        e_ret.append(np.sum(np.abs(np.sqrt(diff) - np.sqrt(diff_ret))**2))
        e_init.append(np.sum(np.abs(np.sqrt(diff) - np.sqrt(diff_init))**2))
        #
        lines_diff.append(np.sum(diff, axis=0))
        lines_ret.append(np.sum(diff_ret, axis=0))
        lines_init.append(np.sum(diff_init, axis=0))
    #
    lines_diff = np.array(lines_diff)
    lines_ret  = np.array(lines_ret)
    lines_init = np.array(lines_init)
    diffs_sum  = np.array(diffs_sum)
    e_ret_line      = np.array(e_ret) / diffs_sum
    e_init_line     = np.array(e_init) / diffs_sum
    #
    #
    plt.clf()
    gs = GridSpec(15, 4)
    gs.update(hspace=0.5)
    #
    ax = plt.subplot(gs[: 2, : 2])
    ax.plot(e_init_line, linewidth=5, alpha=0.5, label='initial')
    ax.plot(e_ret_line , linewidth=5, alpha=0.5, label='retrieved')
    ax.legend()  
    ax.set_title('eMod per diff -- Total(initial, final): '+str(np.sum(e_init)/np.sum(diffs_sum))+','+str(np.sum(e_ret)/np.sum(diffs_sum)), fontsize=18, position=(0.5, 1.01))
    #
    ax = plt.subplot(gs[: 2, 2 :])
    ax.plot(np.log10(eMod), linewidth=5, alpha=0.5, label='modulus')
    ax.plot(np.log10(eSup) , linewidth=5, alpha=0.5, label='support')
    ax.legend()  
    ax.set_title('projection error', fontsize=18, position=(0.5, 1.01))
    #
    ax = plt.subplot(gs[2 : 4 , : 2])
    ax.plot(np.sum(lines_diff, axis=1), linewidth=5, alpha=0.5, label='experiment')
    ax.plot(np.sum(lines_init, axis=1), linewidth=5, alpha=0.5, label='simulation')
    ax.plot(np.sum(lines_ret, axis=1), linewidth=5, alpha=0.5, label='retrieved')
    ax.set_title('Absorption contrast', fontsize=18, position=(0.5, 1.01))
    ax.set_xlabel(r'scan position (pixel)', fontsize=18)
    ax.set_ylabel(r'counts absolute', fontsize=18)
    ax.legend()
    #
    ax = plt.subplot(gs[2 : 4, 2 :])
    ax.plot(100*np.abs(np.sum(lines_diff - lines_init, axis=1))/np.abs(np.sum(lines_diff, axis=1)), linewidth=5, alpha=0.5, label='difference simulation')
    ax.plot(100*np.abs(np.sum(lines_diff - lines_ret, axis=1))/np.abs(np.sum(lines_diff, axis=1)), linewidth=5, alpha=0.5, label='difference retrieved')
    ax.set_title('Difference', fontsize=18, position=(0.5, 1.01))
    ax.set_xlabel(r'scan position (pixel)', fontsize=18)
    ax.set_ylabel(r'counts (%)', fontsize=18)
    ax.legend()
    #
    #
    ax = plt.subplot(gs[6:8,0])
    #ax.imshow(lines_diff[:, 500:1400], aspect='auto', vmax = 8.0e4, vmin=5.0e4)
    ax.imshow(lines_diff, aspect='auto', vmax = 8.0e4, vmin=5.0e4)
    ax.set_title('Data scan=0181 run=0', fontsize=18, position=(0.5, 1.01))
    #
    ax = plt.subplot(gs[6:8,1])
    #ax.imshow(np.roll(lines_init, 0, axis=0)[:, 500:1400], aspect='auto', vmax = 8.0e4, vmin=5.0e4)
    ax.imshow(np.roll(lines_init, 0, axis=0), aspect='auto', vmax = 8.0e4, vmin=5.0e4)
    ax.set_title('Simulation', fontsize=18, position=(0.5, 1.01))
    #
    ax = plt.subplot(gs[6:8,2])
    #ax.imshow(np.abs(lines_diff - lines_init)[:, 500:1400], aspect='auto')
    ax.imshow(np.abs(lines_diff - lines_init), aspect='auto')
    ax.set_title('Difference absolute value', fontsize=18, position=(0.5, 1.01))
    #
    ax = plt.subplot(gs[6:8,3])
    #ax.imshow(np.log(1+np.abs(lines_diff - np.roll(lines_init, 0, axis=0))[:, 500:1400]), aspect='auto')
    ax.imshow(np.log(1+np.abs(lines_diff - np.roll(lines_init, 0, axis=0))), aspect='auto')
    ax.set_title('Difference log', fontsize=18, position=(0.5, 1.01))
    #
    ax = plt.subplot(gs[8:10,0])
    #ax.imshow(lines_diff[:, 500:1400], aspect='auto', vmax = 8.0e4, vmin=5.0e4)
    ax.imshow(lines_diff, aspect='auto', vmax = 8.0e4, vmin=5.0e4)
    ax.set_title('Data scan=0181 run=0', fontsize=18, position=(0.5, 1.01))
    #
    ax = plt.subplot(gs[8:10,1])
    #ax.imshow(np.roll(lines_ret, 0, axis=0)[:, 500:1400], aspect='auto', vmax = 8.0e4, vmin=5.0e4)
    ax.imshow(np.roll(lines_ret, 0, axis=0), aspect='auto', vmax = 8.0e4, vmin=5.0e4)
    ax.set_title('Retrieved', fontsize=18, position=(0.5, 1.01))
    #
    ax = plt.subplot(gs[8:10,2])
    #ax.imshow(np.abs(lines_diff - lines_ret)[:, 500:1400], aspect='auto')
    ax.imshow(np.abs(lines_diff - lines_ret), aspect='auto')
    ax.set_title('Difference absolute value', fontsize=18, position=(0.5, 1.01))
    #
    ax = plt.subplot(gs[8:10,3])
    #ax.imshow(np.log(1+np.abs(lines_diff - np.roll(lines_ret, 0, axis=0))[:, 500:1400]), aspect='auto')
    ax.imshow(np.log(1+np.abs(lines_diff - np.roll(lines_ret, 0, axis=0))), aspect='auto')
    ax.set_title('Difference log', fontsize=18, position=(0.5, 1.01))
    #
    l_coords = [20, 80, 128, 180, 240]
    #
    diffs_exp = []
    for i in range(len(l_coords)):
        diff = np.array(diffs[i], dtype=np.float64)
        diff = diff * mask
        diffs_exp.append(diff[:, :])#diff.shape[1]/2])
    diffs_exp = np.array(diffs_exp)
    #
    diffs_ret = []
    for i in range(len(l_coords)):
        diff_ret   = mask * makeDiff(sample_ret, probe_ret, ij_coords[i])
        diffs_ret.append(diff_ret[:, :])
    diffs_ret = np.array(diffs_ret)
    #
    diffs_init = []
    for i in range(len(l_coords)):
        diff_init   = mask * makeDiff(sample_init, probe_init, ij_coords[i])
        diffs_init.append(diff_init[:, :])
    diffs_init = np.array(diffs_init)
    #
    vmin = np.log(1+diffs_exp).min()
    vmax = np.log(1+diffs_exp).max()
    #    
    for i in range(len(l_coords)):
        ax = plt.subplot(gs[i+10,0])
        ax.imshow(np.log(1+diffs_exp[i]), aspect='auto')#, vmax = vmax, vmin=vmin)
        ax.set_title('pattern '+str(l_coords[i])+' exp log', fontsize=18, position=(0.5, 1.01))
    #
    for i in range(len(l_coords)):
        ax = plt.subplot(gs[i+10,1])
        ax.imshow(np.log(1+diffs_init[i]), aspect='auto')#, vmax = vmax, vmin=vmin)
        ax.set_title('pattern '+str(l_coords[i])+' init log', fontsize=18, position=(0.5, 1.01))
    #
    for i in range(len(l_coords)):
        ax = plt.subplot(gs[i+10,2])
        ax.imshow(np.log(1+diffs_ret[i]), aspect='auto')#, vmax = vmax, vmin=vmin)
        ax.set_title('pattern '+str(l_coords[i])+' ret log', fontsize=18, position=(0.5, 1.01))
    #
    for i in range(len(l_coords)):
        ax = plt.subplot(gs[i+10,3])
        ax.imshow(np.log(1+np.abs(diffs_exp[i]-diffs_ret[i])), aspect='auto')#, vmax = vmax, vmin=vmin)
        ax.set_title('pattern '+str(l_coords[i])+' difference exp / ret log', fontsize=18, position=(0.5, 1.01))
    #
    plt.gcf().set_size_inches(30,50)
    #
    plt.savefig(outputDir + 'fig_errors.png')


if __name__ == '__main__':
    print '#########################################################'
    print 'Processing the results'
    print '#########################################################'
    inputDir, outputDir, scan, run = main(sys.argv[1:])
    print 'input directory is ', inputDir
    print 'output directory is ', outputDir
    #
    print 'Loading the initial and retrieved sample ...'
    sample_ret  = bg.binary_in(inputDir + 'sample_retrieved', dt=np.complex128, dimFnam=True)
    sample_init = bg.binary_in(inputDir + 'sampleInit', dt=np.complex128, dimFnam=True)
    if len(sample_init.shape) == 1 :
        temp = np.zeros_like(sample_ret)
        temp[:] = sample_init
        sample_init = temp
    #
    if fnamBase_match(inputDir + 'sample_support'):
        print 'Loading the sample support...'
        sample_support = bg.binary_in(inputDir + 'sample_support', dt=np.float64, dimFnam=True)
    else :
        sample_support = np.ones_like(sample_init)
    if len(sample_support.shape) == 1 :
        temp = np.zeros_like(sample_ret, dtype=np.float64)
        temp[:] = sample_support
        sample_support = temp
    sample_support = np.array(sample_support, dtype=np.bool)
    #
    print 'Loading the initial and retrieved probe ...'
    probe_ret  = bg.binary_in(inputDir + 'probe_retrieved', dt=np.complex128, dimFnam=True)
    probe_init = bg.binary_in(inputDir + 'probeInit', dt=np.complex128, dimFnam=True)
    #
    print 'Loading the mask...'
    mask = bg.binary_in(inputDir + 'mask', dt=np.float64, dimFnam=True)
    mask = np.array(mask, dtype=np.bool)
    #
    print 'Loading the ij coordinates...'
    coords   = bg.binary_in(inputDir + 'coords_retrieved', dt=np.float64, dimFnam=True)
    print 'warning: casting the coordinates from float to ints.'
    coords = np.array(coords, dtype=np.int32)
    #
    #
    print 'Loading the errors...'
    eMod = bg.binary_in(inputDir + 'error_mod', dt=np.float64, dimFnam=True)
    eSup = bg.binary_in(inputDir + 'error_sup', dt=np.float64, dimFnam=True)
    #
    print 'Making a heatmap on the sample...'
    heatmap = makeHeatmap(probe_init, sample_init, coords)
    #
    print 'Making the sample figure...'
    make_sample_fig(sample_init, sample_ret, sample_support, heatmap, outputDir)
    #
    print 'Making the probe figure...'
    make_probe_fig(probe_init, probe_ret, outputDir, scan, run)
    #
    #print 'Making the errors figure...'
    #make_error_fig(coords, mask, sample_init, sample_ret, sample_support, probe_init, probe_ret, eMod, eSup, inputDir, outputDir)


