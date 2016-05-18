import ConfigParser
import sys, os

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
