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

def load_h5s(fnams):
    diffs = []
    itot  = float(len(fnams))
    for i,fnam in enumerate(fnams):
        update_progress(i/(itot-1.0))
        diffs.append(array_from_h5(fnam))
    return np.array(diffs)

def darkfield_array( scan = '0119', run = 40):
    """I can't find a dark field image at the moment, so I'll skip this for now."""
    zyxN_stack, fnams   = load_metadata(scan = scan)
    zyxN                = zyxN_stack[run]
    dark_diffs          = load_h5s(fnams)
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
    print 'masking the panel edges...'
    mask    = make_panel_edge_mask() 
    print 'masking bad pixels with a mean filter...'
    mask   *= bad_pixels(np.sum(diffs, axis=0))
    print 'masking bad pixels manually...'
    mask[: 427, 784] = 0    # this is a bad streak that occurs often
    mask[: , 829] = 0       # this is a bad pixel that occurs often
    mask[: , 828] = 0    # this is a bad pixel that occurs often
    mask[: , 914] = 0    # this is a bad pixel that occurs often
    mask[: , 785] = 0    # this is a bad pixel that occurs often
    mask[:, 1330 :]  = 0     # mask the plate on the detector 1331 (1 pixel buffer)
    #mask[:, 1490]  = 0     # mask the plate on the detector 1331 (1 pixel buffer)
    print 'Shifting, padding and deleting positive frequencies...'
    mask    = padd_array(mask)
    # Do not allow positive frequencies in the reconstruction
    mask = bg.quadshift(mask)
    mask[:, mask.shape[1]/2 + 10 :] = 1
    mask = bg.iquadshift(mask)
    #
    print 'Shifting, padding and deleting positive frequencies for the diffraction data...'
    diffs_out = []
    itot = float(len(diffs))
    for i, diff in enumerate(diffs):
        update_progress(i/(itot-1.0))
        diffs_out.append(padd_array(diff))
    return np.array(diffs_out), np.array(mask, dtype=np.bool)

############# #############
# Propagation routines
############# #############
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
def _Fresnel_exp(Nx, Ny, Fx, Fy):
    x, y  = bg.make_xy((Nx, Ny))
    exp   = np.exp(-1J * np.pi * (x**2 /(Nx**2*Fx) + y**2 / (Ny**2*Fy)))
    return exp

def Fresnel_exp(shape, Fresnel):
    exp = _Fresnel_exp(shape[0], shape[1], Fresnel[0], Fresnel[1])
    return exp

def prop_Fresnel(psi_in, Fresnel = [np.inf, np.inf]):
    if type(Fresnel) == float :
        Fresnel = [Fresnel, Fresnel]
    #
    if Fresnel == [0, 0] :
        psi_out = psi_in.copy()    # The near-field Fresnel propagator cannot handle this
    elif Fresnel == [np.inf,np.inf] :
        psi_out = psi_in.copy()    # The propagation distance is zero
    else : 
        x, y  = bg.make_xy(psi_in.shape)
        exp   = Fresnel_exp(psi_in.shape, Fresnel)
        psi_out = bg.ifft2( exp * bg.fft2(psi_in))
    return psi_out

def probe_z(probe, zyx_i, spacing, lamb):
    Fresnel      = [spacing[0]**2 / (lamb * zyx_i[0]), spacing[1]**2 / (lamb * zyx_i[0])]
    probe_shift  = prop_Fresnel(probe, Fresnel)
    return probe_shift
############# #############

def make_probe(mask, lamb, dq, scan = '0181'):
    aperture = np.zeros((512, 1536), dtype=np.float64)
    index    = 0
    print 'averaging diffraction data without the grating in it (205, 251)'
    if scan == '0181':
        for i in range(205, 251):
            aperture += array_from_h5(scan = scan, number = i)
            index    += 1
    #
    if scan == '0326':
        print 'making the probe for scan 0326...'
        for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,804,805,806,807,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,838,839,840,841,842,843,844,845,846,847,848,849,850,851,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1407,1408,1409,1410,1411,1412,1413,1414,1415,1416,1417,1418,1419,1420,1421,1422,1423,1424,1425,1426,1427,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440,1441,1442,1443,1444,1445,1446,1447,1448,1608,1609,1610,1611,1612,1613,1614,1615,1616,1617,1618,1619,1620,1621,1622,1623,1624,1625,1626,1627,1628,1629,1630,1631,1632,1633,1634,1635,1636,1637,1809,1810,1811,1812,1813,1814,1815,1816,1817,1818,1819,1820,1821,1822,1823,1824,1825,1826,1827,1828,1829,1830,1831,1832,1833,1834,1835,1836,1837,1838,1839,1840,1841,1842,1843,1844,1845,1846,1847,1848,1849,1850,1851,1852,1853,1854,1855,1856,1857,1858,1859,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035,2036,2037,2038,2039,2040,2041,2042,2043,2044,2045,2046,2047,2048]:
            aperture += array_from_h5(scan = scan, number = i)
            index    += 1
    aperture = aperture / index
    print 'reshaping...'
    aperture = padd_array(aperture)
    #
    # This is a bit awkward but I'm going to fill in the blanks with the data adjascent to 
    # the masked bits... That worked pretty well!
    # star and stop and the left / right boundary of the mask 
    print 'applying the mask...'
    aperture *= mask
    print 'filling in masked pixels "from the left"...'
    for i in range(mask.shape[1]):
        if np.sum(mask, axis = 0)[i] > 0 :
            mstart = i
            break
    for i in range(mask.shape[1]-1, -1, -1):
        if np.sum(mask, axis = 0)[i] > 0 :
            mend = i
            break
    for i in range(mask.shape[0]):
        for j in range(mstart, mend+1):
            if mask[i,j] == 0 :
                # Then keep stepping to the left until
                # a value is found
                for jj in range(j-1, -1, -1):
                    if mask[i, jj] == 1 :
                        aperture[i, j] = aperture[i, jj]
                        break
    #
    # Apply a gaussian mask to the aperture (we can be harsh I think)
    #gaus_mask = np.array(bg.blurthresh_mask(aperture, std=10), dtype=np.float64)
    #gaus_mask = bg.blur(gaus_mask)
    #aperture  *= gaus_mask
    # Let's put some higher order aberrations in there
    C3    = 0.0e-2 
    x, y  = bg.make_xy(aperture.shape)
    exp   = np.exp(-1.0J * np.pi / lamb * C3 * (lamb*dq)**4 * (x**2 + y**2)**2)
    probe = bg.ifft2(np.sqrt(aperture)*exp)
    return probe

def make_sample(probe, coords):
    sample_shape = (probe.shape[0] + np.abs(coords[:, 0].max() - coords[:, 0].min()),  \
                    probe.shape[1] + np.abs(coords[:, 1].max() - coords[:, 1].min()))
    sample = np.ones(sample_shape, dtype=np.complex128)
    return sample

def make_coords(zyx, spacing, shape):
    ij_coords       = np.zeros((len(zyx), 2), dtype=np.int)
    ij_coords[:, 1] = zyx[:, 2] / spacing[1]
    return np.array(ij_coords, dtype=np.int)

def geometry():
    """ """
    # Try: np.exp(-1.0J * np.pi * self.lamb * z * self.dq**2 * (x**2 + y**2))
    du   = 55.0e-6                 # pixel size 
    MLL_detector = 3405.0e-3       # from log book
    E    = 22.0e3 * 1.60217657e-19 # from log book monoenergy
    h    = 6.62606957e-34          # from google
    c    = 299792458               # from google
    lamb = c * h / E               # from brain
    Ax   = 21.0e-6                 # Optical axis to fine edge
    df   = 1.5e-3                  # MLL to focus
    z    = MLL_detector - df       # df / Ax * du * (zero_peak - second_edge)
    dq   = du / (lamb * z)
    X    = 1 / dq
    print 'Experimental geometry:'
    print 'pixel size (um)          ', du*1.0e6
    print 'MLL to detector (m)      ', MLL_detector
    print 'wavelength (A)           ', lamb * 1.0e10
    print 'Energy (Kev)             ', 22.0
    print 'MLL to focus (mm)        ', df * 1.0e3
    print 'field of view (um)       ', X * 1.0e6
    return X, lamb


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
    zyx                     = zyxN[:, : 3]
    fnams                   = fnams_stack[int(run)]
    #
    print 'loading diffraction data...'
    diffs                = load_h5s(fnams)
    #
    print 'Processing diffraction data and generating the diffraction mask...'
    diffs, mask = process_diffs(diffs)
    #
    # Calculate geometry
    print 'calculating the geometry...'
    X, lamb = geometry()
    # 
    # Estimate the in-focus probe
    print 'making the in-focus probe...'
    probe = make_probe(mask, lamb, 1/X)
    # 
    # propagate the probe for the run
    print 'propagating the in-focus probe to the sample plane by (m):', zyx[0][0]
    spacing = [X / probe.shape[0], X / probe.shape[1]]
    probe   = probe_z(probe, zyx[0], spacing, lamb)
    #
    # Convert zyx --> ij coords
    print 'making the ij coordinates...'
    ij_coords = make_coords(zyx, spacing, probe.shape)
    #
    # initialise the sample 
    print 'making the sample shape so that it fits the probe given the positions...'
    sample = make_sample(probe, ij_coords)
    #
    # I want the probe to start at the "right". 0 --> sample.shape[1] - probe.shape[1]
    print 'I want the probe to start at the "right". 0 --> sample.shape[1] - probe.shape[1]'
    ij_coords[:, 1] = ij_coords[:, 1] + (sample.shape[1] - probe.shape[1])
    #
    #
    # Output 
    print """outputing:
        the diffraction data 
        the mask 
        the ij coordinates of the sample
        the probe 
        and the sample
        to""", os.path.abspath(outputdir)
    bg.binary_out(diffs, outputdir + 'diffs', dt=np.float64, appendDim=True) 
    bg.binary_out(mask, outputdir + 'mask', dt=np.float64, appendDim=True) 
    #bg.binary_out(zyxN[:, : 3], outputdir + 'zyx', dt=np.float64, appendDim=True) 
    bg.binary_out(ij_coords, outputdir + 'coords', dt=np.float64, appendDim=True)
    bg.binary_out(sample, outputdir + 'sampleInit', dt=np.complex128, appendDim=True)
    bg.binary_out(probe, outputdir + 'probeInit', dt=np.complex128, appendDim=True)
    print 'done!'



