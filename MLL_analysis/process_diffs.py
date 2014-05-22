# process_diffs.py: (../../../rawdata) .h5  --> (../../../tempdata/MLL_calc) diffs + inital variables 

# Turn h5 files into --> diffs, mask, sampleInit, probeInit and coordsInit

import os, sys, getopt, inspect
import numpy as np
import h5py

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from python_scripts import bagOfns as bg

def main(argv):
    scan = ''
    run = ''
    try :
        opts, args = getopt.getopt(argv,"hs:r:o:",["scan=","run=","outputdir="])
    except getopt.GetoptError:
      print 'process_diffs.py -s <scan> -r <run> -o <outputdir>'
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print 'process_diffs.py -s <scan> -r <run> -o <outputdir>'
         sys.exit()
      elif opt in ("-s", "--scan"):
         scan = arg
      elif opt in ("-r", "--run"):
         run = arg
      elif opt in ("-o", "--outputdir"):
         outputdir = arg
    return scan, run, outputdir

def memoize(f):
    """ Memoization decorator for functions taking one or more arguments. """
    class memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args):
            return self[args]
        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret
    return memodict(f)
@memoize
def get_fnams_faster(dir_base):
    """This makes a list of all of the .h5 files in the base directory. 

    The returned value is a dictionary: e.g. 
    dir_fnam['0098'] = [..., 'dir_base/scan_0098_000134.h5']
    """
    # make a script to scan through data
    dir_scans = []
    dirs = os.listdir(os.path.abspath(dir_base))
    for dirnam in dirs:
        if os.path.isdir(os.path.join(dir_base, dirnam)):
            dir_scans.append( os.path.join(dir_base, dirnam) )
    # now dir_scans is a list of all of the scan directories
    # I want a function for getting the extension number (since the
    # list of scan numbers is not complete or consecutive...
    dir_scans_dict = {}
    for dirnam in dir_scans:
        ending = os.path.basename(dirnam)[-4:]
        dir_scans_dict[ending] = dirnam
    dir_fnam = {}
    for key in dir_scans_dict.keys():
        fnams = [ os.path.join(dir_scans_dict[key], f) for f in os.listdir(dir_scans_dict[key]) if os.path.splitext(f)[-1] == '.h5' ]
        dir_fnam[key] = sorted(fnams)
    return dir_fnam

def load_metadata(path_base = '../../../rawdata/PETRA3/P11/P11-201311/10010762/', scan = '0181'):
    """Returns a list of numpy arrays of z, y, x and N (a unique identifier) coordinates in loaded from the path metadata dir. where each item in the list is a different z-plane along the optical axis. 
    
    In addition to the filename (absolute path) associated with each coordinate set. 
    y_enc is the z (or optical axis direction), and x_enc is the transverse direction"""
    path = path_base + 'metadata/'
    files_temp = sorted(os.listdir(os.path.abspath(os.path.join(path, 'scan_' + scan))))
    files = []
    for fnam in files_temp :
        files.append(os.path.abspath(os.path.join(path, 'scan_' + scan, fnam)))
    # make a list with [pos, x, y] elements
    zxN   = []
    fnams = []
    for file_n in files :
        values = {}
        f = open(file_n, 'r')
        for line in f:
            temp = line.rsplit()
            if len(temp) == 2 :
                try :
                    test = float(temp[1])
                    values[temp[0]] = temp[1]
                except :
                    pass
        try :
            #list_Nxy.append( [file_n, values['pos'], float(values['x_enc']), float(values['y_enc'])])
            zxN.append( [float(values['y_enc']), float(values['x_enc']), values['pos']] )
            fnams.append( file_n )
        except :
            pass
    zyxN       = np.zeros((len(zxN), 4), dtype=np.float64)
    zyxN[:, 0] = np.array(zxN)[:, 0]
    zyxN[:, 2] = np.array(zxN)[:, 1]
    zyxN[:, 3] = np.array(zxN)[:, 2]
    #
    # Now I think that z is in units of mm and pointing upstream of the beam
    zyxN[:, 0] -= zyxN[:, 0].min()
    zyxN[:, 0] *= -1.0e-3
    #
    # I also think there is an offset
    if scan == '0181' :
        zyxN[:, 0] += zyxN[800, 0]
    if scan == '0326' :
        zyxN[:, 0] -= zyxN[500, 0]
    #
    # and I think that x is in units of um
    zyxN[:, 2] *= 1.0e-6
    zyxN[:, 2] -= zyxN[:, 2].min()
    #
    # Replace the file names with the filenames of the associated h5 files
    path     = path_base + 'lambda/'
    dir_fnam = get_fnams_faster(path)
    fnams_h5 = []
    for j in range(len(zyxN)):
        pos = int(zyxN[j, -1])
        # match scan_0119_000123.h5 where pos = 123
        # this is likely to be at or before number = 123
        for i in range(pos, -len(dir_fnam[scan]), -1):
            a = os.path.splitext(dir_fnam[scan][i])[0]
            pos2 = int(a[-6:])
            if pos2 == pos:
                fnams_h5.append(dir_fnam[scan][i])
                break
    #
    # Let's break up the metadata into chunks of z-planes 
    #   This assumes that zyx is sorted in z
    zyxN_stack = []
    zyxN_temp  = []
    zyxN_temp.append(zyxN[0])
    for j in range(1, len(zyxN)):
        if zyxN[j][0] == zyxN[j-1][0] :
            zyxN_temp.append([zyxN[j][0], zyxN[j][1], zyxN[j][2], zyxN[j][3]])
        else :
            zyxN_stack.append(np.array(zyxN_temp))
            zyxN_temp = []
            zyxN_temp.append([zyxN[j][0], zyxN[j][1], zyxN[j][2], zyxN[j][3]])
    zyxN_stack.append(np.array(zyxN_temp))
    # 
    # make an fnams stack as well
    fnams_h5_stack = []
    fnams_h5_temp  = []
    index = 0
    for i in range(len(zyxN_stack)):
        for j in range(len(zyxN_stack[i])):
            fnams_h5_temp.append(fnams_h5[index])
            index += 1
        fnams_h5_stack.append(fnams_h5_temp)
        fnams_h5_temp = []
    return zyxN_stack, fnams_h5_stack

def getfnam_scan_pos(scan = None, number = 0, pos = None):
    dir_fnam = get_fnams_faster('../../../rawdata/PETRA3/P11/P11-201311/10010762/lambda/')
    if pos is None :
        if number == len(dir_fnam[scan]):
            return False
        fnam = dir_fnam[scan][number]
    else :
        # match scan_0119_000123.h5 where pos = 123
        # this is likely to be at or before number = 123
        for i in range(pos, -len(dir_fnam[scan]), -1):
            a = os.path.splitext(dir_fnam[scan][i])[0]
            pos2 = int(a[-6:])
            if pos2 == pos:
                fnam = dir_fnam[scan][i]
    return fnam

def array_from_h5(fnam = None, scan = None, number = 0, pos = None):
    if fnam is None :
        fnam = getfnam_scan_pos(scan = scan, number = number, pos = pos)
    g = h5py.File(fnam)
    path = 'entry/instrument/detector/data'
    array = np.copy(g[path][0])
    h5py.File.close(g)
    return array

def load_h5s(zyxN, fnams):
    diffs = []
    for i in range(len(zyxN)):
        #print i, len(zyxN)
        diff = array_from_h5(fnams[i])
        diffs.append(diff)
    return np.array(diffs)

def darkfield_array( scan = '0119', run = 40):
    """I can't find a dark field image at the moment, so I'll skip this for now."""
    zyxN_stack, fnams   = load_metadata(scan = scan)
    zyxN                = zyxN_stack[run]
    dark_diffs          = load_h5s(zyxN, fnams)
    return np.mean(dark_diffs, axis = 0)

def make_panel_edge_mask(window=2, shape = (512, 1536)):
    ny, nx = shape
    x_edges = np.linspace(0, nx-1, num=7, endpoint=True)
    y_edges = np.linspace(0, ny-1, num=3, endpoint=True)
    edge_mask = np.ones((ny, nx), dtype=np.bool)
    def tempx(i):
        if i <= 0:
            return 0
        elif i >= edge_mask.shape[1]:
            return edge_mask.shape[1]
        else :
            return int(i)
    def tempy(i):
        if i <= 0:
            return 0
        elif i >= edge_mask.shape[0]:
            return edge_mask.shape[0]
        else :
            return int(i)
    window = 2
    for x_edge in x_edges:
        edge_mask[:, tempx(x_edge - window):tempx(x_edge + window)] = 0
    for y_edge in y_edges:
        edge_mask[tempy(y_edge - window):tempy(y_edge + window),:] = 0
    return edge_mask

def bad_pixels(diff, window=2):
    def tempx(i):
            if i <= 0:
                return 0
            elif i >= diff.shape[1]:
                return diff.shape[1]
            else :
                return int(i)
    def tempy(i):
            if i <= 0:
                return 0
            elif i >= diff.shape[0]:
                return diff.shape[0]
            else :
                return int(i)
    def local_window(arrayin, coord = [0,0], window=2):
        return arrayin[ tempy(coord[0] - window): tempy(coord[0] + window),\
                           tempx(coord[1] - window): tempx(coord[1] + window)]
    errant_pixs = []
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            array = local_window(diff, [i,j])
            median = np.median(array)
            if (abs(diff[i,j] - median)  > 20 * abs(median)) and (abs(diff[i,j] - median) > 50):
                errant_pixs.append([i,j])
            elif diff[i,j] <= 0 and np.mean(array) > 20 :
                errant_pixs.append([i,j])
    #
    errant = np.array(errant_pixs)
    errant_mask = np.ones_like(diff, dtype=np.bool)
    errant_mask[errant[:, 0], errant[:, 1]] = False
    return errant_mask

def padd_array(arrayin, zero_pixel = [144, 978]):
    """ """
    #
    # I think we only need to keep the bottom half of the array
    # and let's cut it short in the horizontal direction as well
    arrayout = np.array(arrayin[256 :, 512 :])
    #
    # Now we need to shift and padd, I think we need to have as many positive frequencies as negative?
    # shift right until "zero" is at [0,0]
    arrayout  = bg.roll(arrayout, shift = [-zero_pixel[0], -zero_pixel[1]])
    # Now we double the array width 
    arrayout2 = np.zeros((arrayout.shape[0], 2*arrayout.shape[1]), dtype=arrayout.dtype)
    arrayout2[:, 1024 :] = arrayout 
    return arrayout2

def ipadd_array(arrayout2, zero_pixel = [144, 978]):
    """ """
    arrayout = np.zeros((arrayout2.shape[0], arrayout2.shape[1]/2), dtype=arrayout2.dtype)
    arrayout = arrayout2[:, 1024 :] 
    #
    arrayout  = bg.roll(arrayout, zero_pixel)
    #
    arrayin = np.zeros((512, 1536), dtype=arrayout.dtype)
    arrayin[256 :, 512 :] = np.array(arrayout)
    return arrayin

def process_diffs(diffs):
    """Process the diffraction data and return diffs, mask:

    Padd the diffraction patterns
    Padd the mask
    make the mask by masking edges and finding bad pixels"""
    mask    = make_panel_edge_mask() 
    mask   *= bad_pixels(np.sum(diffs, axis=0))
    mask[: 427, 784] = 0    # this is a bad streak that occurs often
    mask[: , 829] = 0       # this is a bad pixel that occurs often
    mask[: , 828] = 0    # this is a bad pixel that occurs often
    mask[: , 914] = 0    # this is a bad pixel that occurs often
    mask[: , 785] = 0    # this is a bad pixel that occurs often
    mask[:, 1330 :]  = 0     # mask the plate on the detector 1331 (1 pixel buffer)
    #mask[:, 1490]  = 0     # mask the plate on the detector 1331 (1 pixel buffer)
    mask    = padd_array(mask)
    # Do not allow positive frequencies in the reconstruction
    mask = bg.quadshift(mask)
    mask[:, mask.shape[1]/2 + 10 :] = 1
    mask = bg.iquadshift(mask)
    #
    diffs_out = []
    for i in range(len(diffs)):
        print i, len(diffs)
        diffs_out.append(padd_array(diffs[i]))
    return np.array(diffs_out), mask

if __name__ == "__main__":
    print '#########################################################'
    print 'Processing diffraction data'
    print '#########################################################'
    scan, run, outputdir = main(sys.argv[1:])
    print 'scan number is ', scan
    print 'run is ', run
    print 'output directory is ', outputdir
    #
    print 'loading metadata...'
    zyxN_stack, fnams_stack = load_metadata(scan = scan)
    zyxN                    = zyxN_stack[int(run)]
    fnams                   = fnams_stack[int(run)]
    #
    print 'loading diffraction data...'
    diffs                = load_h5s(zyxN, fnams)
    #
    print 'Processing diffraction data and generating the diffraction mask...'
    diffs, mask = process_diffs(diffs)
    #
    #
    # Output 
    bg.binary_out(diffs, outputdir + 'diffs', dt=np.float64, appendDim=True) 
    bg.binary_out(mask, outputdir + 'mask', dt=np.float64, appendDim=True) 
    #bg.binary_out(zyxN[:, : 3], outputdir + 'zyx', dt=np.float64, appendDim=True) 




