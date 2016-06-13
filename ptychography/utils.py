import ConfigParser
import sys, os
import numpy as np

def parse_cmdline_args_cxi_mask_editor():
    import argparse
    import os
    parser = argparse.ArgumentParser(prog = 'python cxi_mask_editor.py', description='edit the hot/dead or pupil mask of an mll-cxi file')
    parser.add_argument('config', type=str, \
                        help="configuration file name")
    parser.add_argument('mask_type', type=str, \
                        help="on of 'hot', 'dead', 'pupil'")

    args = parser.parse_args()

    # check that args.config exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)

    # process config file
    config = ConfigParser.ConfigParser()
    config.read(args.config)
    
    params = parse_parameters(config)
     
    params['mask_type'] = args.mask_type
    return params

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(prog = 'mpirun -n N python mll_cxi_wrapper.py', description='')
    parser.add_argument('config', type=str, \
                        help="configuration file name")

    args = parser.parse_args()

    # check that args.config exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)

    # process config file
    config = ConfigParser.ConfigParser()
    config.read(args.config)
    
    params = parse_parameters(config)
    return params

def parse_parameters(config):
    """
    Parse values from the configuration file and sets internal parameter accordingly
    The parameter dictionary is made available to both the workers and the master nodes
    The parser tries to interpret an entry in the configuration file as follows:
    - If the entry starts and ends with a single quote, it is interpreted as a string
    - If the entry is the word None, without quotes, then the entry is interpreted as NoneType
    - If the entry is the word False, without quotes, then the entry is interpreted as a boolean False
    - If the entry is the word True, without quotes, then the entry is interpreted as a boolean True
    - If non of the previous options match the content of the entry, the parser tries to interpret the entry in order as:
        - An integer number
        - A float number
        - A string
      The first choice that succeeds determines the entry type
    """

    monitor_params = {}

    for sect in config.sections():
        monitor_params[sect]={}
        for op in config.options(sect):
            monitor_params[sect][op] = config.get(sect, op)
            if monitor_params[sect][op].startswith("'") and monitor_params[sect][op].endswith("'"):
                monitor_params[sect][op] = monitor_params[sect][op][1:-1]
                continue
            if monitor_params[sect][op] == 'None':
                monitor_params[sect][op] = None
                continue
            if monitor_params[sect][op] == 'False':
                monitor_params[sect][op] = False
                continue
            if monitor_params[sect][op] == 'True':
                monitor_params[sect][op] = True
                continue
            try:
                monitor_params[sect][op] = int(monitor_params[sect][op])
                continue
            except :
                try :
                    monitor_params[sect][op] = float(monitor_params[sect][op])
                    continue
                except :
                    # attempt to pass as an array of ints e.g. '1, 2, 3'
                    try :
                        l = monitor_params[sect][op].split(',')
                        temp = int(l[0])
                        monitor_params[sect][op] = np.array(l, dtype=np.int)
                        continue
                    except :
                        try :
                            l = monitor_params[sect][op].split(',')
                            temp = float(l[0])
                            monitor_params[sect][op] = np.array(l, dtype=np.float)
                            continue
                        except :
                            try :
                                l = monitor_params[sect][op].split(',')
                                if len(l) > 1 :
                                    monitor_params[sect][op] = [i.strip() for i in l]
                                continue
                            except :
                                pass

    return monitor_params

def crop_to_nonzero(arrayin, mask=None):
    """Crop arrayin to the smallest rectangle that contains all of the non-zero elements and return the result. If mask is given use that to determine non-zero elements.
    
    If arrayin is a list of arrays then all arrays are cropped according to the first in the list."""

    if type(arrayin) == np.ndarray :
        array = arrayin
    elif type(arrayin) == list :
        array = arrayin[0]

    if mask==None :
        mask = array
    #most left point 
    for i in range(mask.shape[1]):
        tot = np.sum(np.abs(mask[:,i]))
        if tot > 0.0 :
            break
    left = i
    #most right point 
    for i in range(mask.shape[1]-1,-1,-1):
        tot = np.sum(np.abs(mask[:,i]))
        if tot > 0.0 :
            break
    right = i
    #most up point 
    for i in range(mask.shape[0]):
        tot = np.sum(np.abs(mask[i,:]))
        if tot > 0.0 :
            break
    top = i
    #most down point
    for i in range(mask.shape[0]-1,-1,-1):
        tot = np.sum(np.abs(mask[i,:]))
        if tot > 0.0 :
            break
    bottom = i
    if type(arrayin) == np.ndarray :
        arrayout = array[top:bottom+1,left:right+1]
    elif type(arrayin) == list :
        arrayout = []
        for i in arrayin :
            arrayout.append(i[top:bottom+1,left:right+1])
    return arrayout
