
import h5py
import sys, os
import numpy as np

def parse_cmdline_args_cxi_mask_editor():
    import argparse
    import os
    parser = argparse.ArgumentParser(prog = 'python cxi_mask_editor.py', description='edit the hot/dead or pupil mask of an mll-cxi file')
    parser.add_argument('cxi_fnam', type=str, \
                        help="filename of mll cxi file")
    parser.add_argument('mask_type', type=str, \
                        help="on of 'hot', 'dead', 'pupil'")

    args = parser.parse_args()
     
    params = {}
    params['mask_type'] = args.mask_type
    params['input'] = {'cxi_fnam': args.cxi_fnam}
    return params

def if_exists_del(fnam):
    import os
    # check that the directory exists and is a directory
    output_dir = os.path.split( os.path.realpath(fnam) )[0]
    if os.path.exists(output_dir) == False :
        raise ValueError('specified path does not exist: ', output_dir)
    
    if os.path.isdir(output_dir) == False :
        raise ValueError('specified path is not a path you dummy: ', output_dir)
    
    # see if it exists and if so delete it 
    # (probably dangerous but otherwise this gets really anoying for debuging)
    if os.path.exists(fnam):
        print '\n', fnam ,'file already exists, deleting the old one and making a new one'
        os.remove(fnam)

def launch_pupil_maker(params):
    print params
    # load the cxi file
    f = h5py.File(params['input']['cxi_fnam'], 'r')

    import os
    # get the data
    if 'process_2' in f.keys() and 'powder' in f['process_2'].keys():
        sum = f['process_2/powder'].value
    elif f['entry_1/data_1/data'].shape[0] > 20 :
        sum = np.sum(f['entry_1/data_1/data'][:20], axis=0)
    else :
        sum = np.sum(f['entry_1/data_1/data'].value, axis=0)
    mask = f['entry_1/instrument_1/detector_1/mask'].value
    f.close()
    
    g = h5py.File('sum.h5', 'w')
    g.create_dataset('data', data = sum)
    g.close()

    mask_hot   = (mask & 2**2) > 0
    mask_dead  = (mask & 2**3) > 0
    mask_pupil = (mask & 2**10) > 0
    
    g = h5py.File('new_mask.h5', 'w')
    if params['mask_type'] == 'hot' :
        print 'writing existing hot mask to new_mask.h5'
        g.create_dataset('data', data = (1-mask_hot).astype(np.int16))
    elif params['mask_type'] == 'dead' :
        print 'writing existing dead mask to new_mask.h5'
        g.create_dataset('data', data = (1-mask_dead).astype(np.int16))
    elif params['mask_type'] == 'pupil' :
        print 'writing existing pupil mask to new_mask.h5'
        g.create_dataset('data', data = (1-mask_pupil).astype(np.int16))
    g.close()
    
    if_exists_del('mask.h5')
    
    # pupil mask
    print '\nfill in the', params['mask_type'], 'mask (in blue):'
    os.system('./CsPadMaskMaker/maskMakerGUI.py sum.h5 data -m new_mask.h5 -mp data')
    print '\nLoading the', params['mask_type'], 'mask'
    new_mask = 1-h5py.File('mask.h5', 'r')['data/data'].value

    # first set the bits to zero
    if params['mask_type'] == 'hot' :
        mask = mask - (mask & 2**2)
        print 'making hot pixel mask'
        mask += 2**2 * new_mask
    elif params['mask_type'] == 'dead' :
        mask = mask - (mask & 2**3)
        print 'making dead pixel mask'
        mask += 2**3 * new_mask
    elif params['mask_type'] == 'pupil' :
        mask = mask - (mask & 2**10)
        print 'making pupil pixel mask'
        mask += 2**10 * new_mask

    print 'writing to cxi file...'
    f = h5py.File(params['input']['cxi_fnam'], 'a')
    f['entry_1/instrument_1/detector_1/mask'][:] = mask
    f.close()
    if_exists_del('mask.h5')
    if_exists_del('new_mask.h5')
    if_exists_del('sum.h5')
    print 'Done!'

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

def read_frames_with_pupil(cxi_file, no_frames = None):
    mask  = cxi_file['entry_1/instrument_1/detector_1/mask'].value
    pupil = np.bitwise_and(mask, 2**10) > 0
    
    dset = cxi_file['entry_1/data_1/data']
    if no_frames is None :
        no_frames = dset.shape[0]
    
    a = crop_to_nonzero(pupil)
    
    frames = np.zeros((no_frames,) + a.shape, dtype = dset.dtype)

    for i in range(len(frames)):
        frames[i][a > 0] = dset[i][pupil > 0]
    return frames

if __name__ == '__main__':
    params = parse_cmdline_args_cxi_mask_editor()
    launch_pupil_maker(params)
    
