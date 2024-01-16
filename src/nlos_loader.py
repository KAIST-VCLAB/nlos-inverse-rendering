import numpy as np
from scipy.io import loadmat
from easydict import EasyDict as edict
import mat73
import cv2

import warnings
warnings.filterwarnings("ignore")

def reduce_data(data_txhxw, bin_len, minimal_wallpos, maximal_wallpos, sampling_grid_spacing, compress_step=0, reduce_tnum_only=False):

    for k in range(compress_step):
        if reduce_tnum_only == False:
            data_txhxw = (data_txhxw[:, ::2, :] + data_txhxw[:, 1::2, :])
            data_txhxw = (data_txhxw[:, :, ::2] + data_txhxw[:, :, 1::2])

            # minor manipulation of boundary positions
            minimal_wallpos[:2] += (sampling_grid_spacing * 0.5)
            maximal_wallpos[:2] -= (sampling_grid_spacing * 0.5)
            minimal_wallpos[2] += (bin_len * 0.5)
            maximal_wallpos[2] -= (bin_len * 0.5)
            sampling_grid_spacing = sampling_grid_spacing * 2

        data_txhxw = (data_txhxw[::2, :, :] + data_txhxw[1::2, :, :])
        bin_len = bin_len * 2

    return data_txhxw, bin_len, minimal_wallpos, maximal_wallpos, sampling_grid_spacing


def calculate_relay_wall_positions(hnum, wnum, minimal_wallpos, maximal_wallpos, relayWallDepth, isConfocal=True, spad_index=None):

    lx_axis = np.linspace(minimal_wallpos[0], maximal_wallpos[0], wnum)
    ly_axis = np.linspace(maximal_wallpos[1], minimal_wallpos[1], hnum)
    [ly, lx] = np.meshgrid(ly_axis, lx_axis, indexing='ij')
    lz = np.zeros([hnum, wnum]) + relayWallDepth
    laserPos = np.stack([lx, ly, lz], axis=2)

    if isConfocal:
        cameraPos = laserPos
    else:
        if spad_index is None:
            # assumption: the camera is located at the center of the relay wall
            cameraPos = np.array([(minimal_wallpos[0] + maximal_wallpos[0]) * 0.5,
                            (minimal_wallpos[1] + maximal_wallpos[1]) * 0.5, relayWallDepth])
        else:
            h = spad_index[0]
            w = spad_index[1]
            cameraPos = np.array([laserPos[h,w,0], laserPos[h,w,1], relayWallDepth])
        cameraPos = np.expand_dims(cameraPos, axis=0)
        cameraPos = np.tile(cameraPos, [hnum, wnum, 1])

    # result shape : hnum x wnum x 3
    return laserPos, cameraPos


def read_nlos(filename, 
              data_type=0,
              isConfocal=True, 
              bin_len_sec=4e-12, 
              relay_wall_h=2.0,
              relay_wall_w=2.0,
              compress_step=0, 
              depth_min=1., 
              depth_max=2.,
              v_apt_Sz=None
              ):
    
    c_light = 299792458
    relayWallDepth = 0.0
    bin_len = bin_len_sec * c_light    

    data_dict = edict()

    reduce_tnum_only = False
    spad_index = None
    data_dict.cycle_times = 4
    if data_type==0:
        # Chen et al. renderer data converted to mat file
        data_bundle = loadmat(filename)
        data_txhxw = data_bundle['meas'] 
        rect_data_txhxw = data_txhxw
    elif data_type==1:
        # Z-NLOS data
        data_bundle = loadmat(filename)
        rect_data = data_bundle['rect_data']
        rect_data_txhxw = rect_data.transpose((2,1,0))
        rect_data_txhxw = np.flip(rect_data_txhxw, axis=1)
        rect_data_txhxw = np.flip(rect_data_txhxw, axis=2)            
    elif data_type==2:
        # LCT data
        data_bundle = loadmat(filename)                
        rect_data = data_bundle['rect_data']
        rect_data_txhxw = rect_data.transpose((2,1,0))      
        rect_data_txhxw[:600, :, :] = 0.
        rect_data_txhxw = np.flip(rect_data_txhxw, axis=2)
    elif data_type==3:
        # Phasor-field data
        data_bundle = mat73.loadmat(filename) 
        rect_data = data_bundle['rect_data']
        rect_data_txhxw = rect_data.transpose((2,0,1))
        rect_data_txhxw[:500, :, :] = 0.
        rect_data_txhxw = np.flip(rect_data_txhxw, axis=2)
        valid_tnum = round(depth_max * 2 / bin_len)
        for i in range(15):
            tgt_tnum = 2 ** i
            if valid_tnum < tgt_tnum:
                tnum = tgt_tnum
                break    
        rect_data_txhxw =  rect_data_txhxw[:tnum, :, :]
        spad_index = np.array(data_bundle['SPAD_index']).astype(np.int64)
        data_dict.cycle_times = 5
        reduce_tnum_only = True
    elif data_type==4:
        # F-K migration data
        data_bundle = mat73.loadmat(filename)
        rect_data = data_bundle['meas']
        rect_data_txhxw = rect_data.transpose((2,0,1)) 
    elif data_type==5:
        # Tsai et al. data
        data_bundle = loadmat(filename)
        rect_data = data_bundle['meas']
        rect_data_txhxw = rect_data.transpose((2,1,0)) 
        rect_data_txhxw = np.flip(rect_data_txhxw, axis=1)
    elif data_type==6:
        # Chen et al. renderer data
        im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)        
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        imgray = imgray.reshape([1200, 256, 256])
        rect_data_txhxw = np.array(imgray, dtype=np.float32)
        rect_data_txhxw = rect_data_txhxw[:1024, :, :]
    else:
        assert False, "Unreachable!"
    
    tnum, hnum, wnum = rect_data_txhxw.shape
    data_txhxw = rect_data_txhxw

    minimal_wallpos = np.array([-relay_wall_w/2., -relay_wall_h/2., 0.0])
    maximal_wallpos = np.array([relay_wall_w/2., relay_wall_h/2., 0.0])
    sampling_grid_spacing = relay_wall_w / (wnum - 1)     # assumption: vertical, horizontal spacing is the same

    data_txhxw, bin_len, minimal_wallpos, maximal_wallpos, sampling_grid_spacing \
        = reduce_data(data_txhxw, bin_len, minimal_wallpos, maximal_wallpos, sampling_grid_spacing, compress_step, reduce_tnum_only)
    tnum, hnum, wnum = data_txhxw.shape
    tmeas_max = np.max(data_txhxw)
    data_txhxw = data_txhxw / tmeas_max     # normalize for optimizing

    laserPos, cameraPos = calculate_relay_wall_positions(
        hnum, wnum, minimal_wallpos, maximal_wallpos, relayWallDepth, isConfocal, spad_index)

    # case for bunny
    minimalpos = np.array([minimal_wallpos[0], minimal_wallpos[1], depth_min])
    maximalpos = np.array([maximal_wallpos[0], maximal_wallpos[1], depth_max])

    data_dict.tnum = tnum               # number of time bins
    data_dict.hnum = hnum               # number of height relay wall points
    data_dict.wnum = wnum               # number of width relay wall points
    data_dict.tmeas = data_txhxw.astype(np.float32)        # transient measurements of tnum x hnum x wnum
                                        # laser_pos[y, x] and camera_pos[y, x] corresponds to data_txhxw[:,y,x]
    data_dict.bin_len = bin_len         # time bin resolution represented in a length [m]
    data_dict.sampling_grid_spacing = sampling_grid_spacing
    data_dict.min_pos = minimalpos.astype(np.float32)      # boundary coordinates of object space [m]
    data_dict.max_pos = maximalpos.astype(np.float32)      # boundary coordinates of object space [m]
    data_dict.laser_pos = laserPos.astype(np.float32)      # laser coordinates of relay wall [m]
    data_dict.camera_pos = cameraPos.astype(np.float32)    # camera_coordinates of relay wall [m]
    data_dict.laser_origin = data_dict.laser_pos.copy()    # for emulate 3-bounce model
    data_dict.camera_origin = data_dict.camera_pos.copy()   # for emulate 3-bounce model
    data_dict.lambda_times = 2
    data_dict.c_light = c_light
    data_dict.isConfocal = isConfocal
    if v_apt_Sz is None:
        data_dict.virtual_aperture_size = maximal_wallpos[0] - minimal_wallpos[0]
    else:
        data_dict.virtual_aperture_size = v_apt_Sz

    return data_dict