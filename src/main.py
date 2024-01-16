import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import (
    RayBundle,
    ray_bundle_to_ray_points,
    get_tmeas_scale,
    get_figures,
    penalize_min_max,
)
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict as edict
from pathlib import Path
import argparse
import json

from nlos_loader import read_nlos
from phasor_field_imager import PhasorFieldImager
from hemisphere_raysampler import HemisphereRaysampler
from hemisphere_raymarcher import HemisphereRaymarcher
from point_transient_renderer import PointTransientRenderer
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import datetime
import open3d as o3d


class SelfSupervisedOptimizer(nn.Module):
    def __init__(self, 
        data_dict, 
        threshold_intensity = 0.01,
        init_with_original_kernel = False,
        freq_band_enable = False,        
        volume_albedo_init = None,
        depth_mode = 0,
        device = 'cuda:0'):
        super().__init__()

        self.args = data_dict
        self.device = device
        self.phasor_field_imager = PhasorFieldImager(
                args = data_dict, 
                init_with_original_kernel = init_with_original_kernel,
                init_scale_sigma = data_dict.init_scale_sigma,
                init_scale_omega = data_dict.init_scale_omega,
                freq_band_enable = freq_band_enable,                
                depth_mode = depth_mode,
                device = device
                )
        self.ray_marcher = HemisphereRaymarcher(
                args = data_dict, 
                max_lengths_from_spot = 1e3,
                threshold_intensity = threshold_intensity,
                device = device
                )
        self.point_transient_renderer = PointTransientRenderer(
                args = data_dict, 
                volume_albedo_init = volume_albedo_init,
                threshold_intensity = threshold_intensity,
                device = device
                )
        self.tmeas = None

    
    def forward(self, tmeas, ray_bundle:RayBundle):
        ray_points = ray_bundle_to_ray_points(ray_bundle)
        self.tmeas = tmeas
        ray_intensities, phasor_volume = self.phasor_field_imager(self.tmeas, ray_points)
        surf_pos, surf_normals = self.ray_marcher(ray_bundle, ray_intensities)
        tdata_out, tdata = self.point_transient_renderer(surf_pos, surf_normals, ray_bundle.xys, phasor_volume)

        return tdata_out, surf_pos, surf_normals, tdata    # [mc_size, tnum]


    def clip_parameters(self):
        for k, v in self.named_parameters():
            if k == 'phasor_field_imager.params':
                v.data[0].clamp_(2e0, 1e3)    # sigma_inv
                v.data[1].clamp_(1e-5, 5e2)    # omega
            elif k == 'point_transient_renderer.param_scale':
                v.data.clamp_(1e-5, None)
            elif k == 'point_transient_renderer.param_sigma':
                v.data.clamp_(1e-5, None)
            elif k == 'point_transient_renderer.param_lamda':
                v.data.clamp_(1e-5, None)
            elif k == 'point_transient_renderer.param_dc':
                v.data.clamp_(1e-5, None)
            elif k == 'point_transient_renderer.params_albedo':
                v.data.clamp_(-0.2, 0.2)            
            else:
                pass


    def get_transients(self, idx_h, idx_w):
        # [H, W, T]
        return self.tmeas[idx_h, idx_w, :]



if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    default_file_name = './config/default_scenario.json'
    parser.add_argument('--config_path', type=str, default=default_file_name)
    _args = parser.parse_args()
    args = edict()
    try:
        f = open(_args.config_path, 'r')
        valid_file_path = _args.config_path        
    except:    
        f = open(default_file_name, 'r')
        valid_file_path = default_file_name
    head, tail = os.path.split(valid_file_path)
    args.name = tail[:-5]
    cfg = json.load(f)
    print(f'Configuration file \'{args.name}.json\' is loaded!')

    # load the values from json to args
    for k, v in cfg.items():
        setattr(args, k, v)

    device = f'cuda:{args.rank}'
    n_spots_per_relay_wall = args.spot_count  # should be configured for batch size
    n_rays_per_spot = args.rays_per_spot   # should be a squared number (e.g. 100, 225, 400, ...)
    result_dir = Path(f'./results_{args.name}')
    log_dir = result_dir / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = result_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = log_dir    
    pcd_dir =  result_dir / 'pcd'
    pcd_dir.mkdir(parents=True, exist_ok=True)

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.deterministic = True

    data_dict = read_nlos(args.data_path, 
                        data_type = args.data_type,
                        isConfocal = args.is_confocal, 
                        bin_len_sec = args.bin_len_sec,
                        relay_wall_h = args.relay_wall_h,
                        relay_wall_w = args.relay_wall_w,                        
                        compress_step = args.compress_step,
                        depth_min = args.depth_min,
                        depth_max = args.depth_max,
                        v_apt_Sz = args.v_apt_Sz
                        )

    if args.tensorboard:
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    torch.set_num_threads(1)

    tmeas = torch.from_numpy(data_dict.tmeas[None]).to(device)  # [B, T, H, W]
    tmeas = tmeas.squeeze().permute(1, 2, 0)  # [H, W, T]
    # remove the outlier signals and scaling max to 1.0
    tmeas_scale = get_tmeas_scale(tmeas, 0.999999)
    tmeas = tmeas / tmeas_scale
    tmeas[tmeas > 1.0] = 1.0    

    data_dict.tmeas = tmeas
    data_dict.init_scale_sigma = args.init_scale_sigma
    data_dict.init_scale_omega = args.init_scale_omega
    data_dict.laser_pulse_width = 40e-12
    data_dict.response_lambda = 5.0
    data_dict.response_dc = 1e-3

    L_i_init = None
    volume_albedo_init = None
    # start ######################################## just for inital L_i, albedo_estimate
    initiator_for_sampler = HemisphereRaysampler(
        args = data_dict,
        n_rays_per_spot = n_rays_per_spot,
        n_spots_per_relay_wall = 8, # Monte-Carlo number
        device = device
        )
    initiator_for_pipeline = SelfSupervisedOptimizer(
            data_dict = data_dict, 
            threshold_intensity = args.threshold_intensity,
            init_with_original_kernel = True,
            depth_mode=args.depth_mode,
            device = device
            )             
    with torch.no_grad():
        initiator_for_pipeline = initiator_for_pipeline.to(device)
        r_bundle = initiator_for_sampler()        
        h = r_bundle.xys[:,0,0]
        w = r_bundle.xys[:,0,1]
        tdata_init, _, _, _ = initiator_for_pipeline(tmeas, r_bundle)
        center_index = 0
        L_i_init = tmeas[h[center_index], w[center_index], :].max() / tdata_init[center_index,:].max()
        volume_albedo_init = initiator_for_pipeline.phasor_field_imager.get_volume()
        volume_albedo_init[volume_albedo_init < args.threshold_intensity] = 0.
        volume_albedo_init[volume_albedo_init >= args.threshold_intensity] = 1.0
    del initiator_for_pipeline, initiator_for_sampler    
    # end ######################################## 

    ray_sampler = HemisphereRaysampler(
            args = data_dict,
            n_rays_per_spot = n_rays_per_spot,
            n_spots_per_relay_wall = n_spots_per_relay_wall, # Monte-Carlo number
            device = device
            )
    self_supervised_optimizer = SelfSupervisedOptimizer(
            data_dict = data_dict, 
            threshold_intensity = args.threshold_intensity,
            init_with_original_kernel = args.init_with_original_kernel,
            freq_band_enable = True,
            volume_albedo_init=volume_albedo_init,
            depth_mode=args.depth_mode,
            device = device
            )   

    # prepare for the transient measurements
    self_supervised_optimizer = self_supervised_optimizer.to(device)
    self_supervised_optimizer.point_transient_renderer.param_scale.data = L_i_init

    # employ the laser sensor model to the synthetic data
    if args.data_type == 0 or args.data_type == 1 or args.data_type == 6:
        with torch.no_grad():
            tmeas_reshape = tmeas.reshape((-1, data_dict.tnum))
            tdata_out = self_supervised_optimizer.point_transient_renderer.make_tmeas_to_realistic(tmeas_reshape)
            tmeas = tdata_out.reshape((data_dict.hnum, data_dict.wnum, data_dict.tnum))

    total_params = []
    render_params = dict(self_supervised_optimizer.point_transient_renderer.named_parameters())
    kernel_params = dict(self_supervised_optimizer.phasor_field_imager.named_parameters())
    param_group_index = 0

    total_params.append(render_params['param_scale'])        # param_scale
    optimizer = optim.Adam(total_params, lr=args.lr_li)
    param_group_index = param_group_index + 1

    optimizer.add_param_group({'params': kernel_params['params']})
    param_grounp_index_kernel = param_group_index
    optimizer.param_groups[param_group_index]['lr'] = args.lr
    param_group_index = param_group_index + 1

    optimizer.add_param_group({'params': render_params['params_albedo']})
    optimizer.param_groups[param_group_index]['lr'] = args.lr_albedo
    param_group_index = param_group_index + 1

    optimizer.add_param_group({'params': render_params['param_lamda']})
    optimizer.add_param_group({'params': render_params['param_dc']})    
    optimizer.add_param_group({'params': render_params['param_sigma']})    
    optimizer.param_groups[param_group_index]['lr'] = args.lr_sensor_lamda
    param_group_index = param_group_index + 1
    optimizer.param_groups[param_group_index]['lr'] = args.lr_ambient_dc
    param_group_index = param_group_index + 1
    optimizer.param_groups[param_group_index]['lr'] = args.lr_laser_sigma
    param_group_index = param_group_index + 1

    begin_idx = torch.argmax((tmeas > 0.0).to(dtype=torch.int), dim=-1)
    axis = torch.arange(data_dict.tnum)
    axis_tensor = torch.tile(axis, (data_dict.hnum, data_dict.wnum, 1)).to(device)
    _valid_mask = axis_tensor - begin_idx[..., None]
    valid_mask = torch.where(_valid_mask < 0, 0, 1).float().to(device)

    sigma_min, sigma_max, lamda_min, lamda_max = self_supervised_optimizer.point_transient_renderer.get_params_min_max()
    best_loss = np.inf
    total_start_time = time.time()
    log_iter = args.log_iter
    lr_kernel = args.lr

    print('Optimization started!')

    for iter in range(args.max_iter):

        if iter <= 10:
            lambda_volume_tv = args.lambda_volume_tv * ((1e2 - 1) * np.sin(2*np.pi/40 * iter) + 1)
            optimizer.param_groups[param_grounp_index_kernel]['lr'] = lr_kernel
            lr_kernel = lr_kernel + 1
        else:
            lambda_volume_tv = (10 ** 4.75) * ((10 ** 3.75) ** (-iter / 50))

        # generate rays for random spots of a relay wall
        ray_bundle = ray_sampler()        
        idx_h = ray_bundle.xys[:,0,0]
        idx_w = ray_bundle.xys[:,0,1]

        optimizer.zero_grad()

        # call the main optimization pipeline
        recon_tdata, surf_center_pos, surf_normals, recon_tdata_no_ls = self_supervised_optimizer(tmeas, ray_bundle)

        origin_tdata = tmeas[idx_h, idx_w, :]
        ref_tdata = self_supervised_optimizer.get_transients(idx_h, idx_w)
       
        # compute a loss
        volume_tv = self_supervised_optimizer.phasor_field_imager.get_volume_tv()
        albedo_tv = self_supervised_optimizer.point_transient_renderer.get_albedo_tv()
        tv_term = (lambda_volume_tv * volume_tv) + (args.lambda_albedo_tv * albedo_tv)        
        penalty_term = args.lambda_neg_penalty * (
                    penalize_min_max(self_supervised_optimizer.point_transient_renderer.param_sigma[0], sigma_min, sigma_max)
                    + penalize_min_max(self_supervised_optimizer.point_transient_renderer.param_lamda[0], lamda_min, lamda_max)
                    ) 
        _valid_mask_tdata = valid_mask[idx_h, idx_w, :]
        valid_mask_tdata = _valid_mask_tdata[1:, :]     # remove the first position (it is reserved as a center position for tmeas comparison figure)
        mse_loss_term = F.mse_loss(recon_tdata[1:, :] * valid_mask_tdata, ref_tdata[1:, :] * valid_mask_tdata) * args.lambda_mse_loss
        loss = mse_loss_term + tv_term + penalty_term

        # write logs
        if iter % log_iter == 0 and writer is not None:
            
            writer.add_scalar('loss', loss, iter)
            writer.add_scalar('transient_loss', mse_loss_term, iter)

            volume = self_supervised_optimizer.phasor_field_imager.get_volume_np()
            albedo_volume = self_supervised_optimizer.point_transient_renderer.get_albedo_np()
            writer.add_figure('volume intensity', get_figures(volume), iter)
            writer.add_figure('albedo', get_figures(albedo_volume), iter)
            img_path = raw_dir / f'recon{iter:06d}.exr'
            cv2.imwrite(str(img_path), volume.max(axis=-1))
            
            writer.add_scalar('kernel_sigma', 1. / self_supervised_optimizer.phasor_field_imager.params[0].data, iter)
            writer.add_scalar('kernel_omega', self_supervised_optimizer.phasor_field_imager.params[1].data, iter)
            writer.add_scalar('L_i', self_supervised_optimizer.point_transient_renderer.param_scale.data, iter)

            laser_sigma = self_supervised_optimizer.point_transient_renderer.param_sigma.data
            writer.add_scalar('ls_sigma', self_supervised_optimizer.point_transient_renderer.param_sigma.data, iter)
            writer.add_scalar('ls_kappa', self_supervised_optimizer.point_transient_renderer.param_lamda.data, iter)
            writer.add_scalar('ls_eta', self_supervised_optimizer.point_transient_renderer.param_dc.data, iter)
           
            center = (self_supervised_optimizer.ray_marcher.min_pos + self_supervised_optimizer.ray_marcher.max_pos) / 2
            surf_normals_np = surf_normals.reshape(-1,3).cpu().detach().numpy()
            surf_center_pos_np = surf_center_pos.reshape(-1, 3).cpu().detach().numpy()
            surf_center_pos_np[..., 2] = -surf_center_pos_np[..., 2]
            surf_normals_np[..., 2] = -surf_normals_np[..., 2]
            center[..., 2] = -center[..., 2]

            from numpy import linalg as LA
            dist = LA.norm(surf_center_pos_np - center[None], axis=1)
            surf_center_pos_np = surf_center_pos_np[dist <= 5.0, :]
            surf_normals_np = surf_normals_np[dist <= 5.0, :]
            color = (surf_normals_np + 1) / 2

            X = surf_center_pos_np[:,0]
            Y = surf_center_pos_np[:,1]
            Z = surf_center_pos_np[:,2]

            fig100 = plt.figure(dpi=300)
            ax = fig100.gca(projection = '3d')        
            ax.scatter(X, Z, Y, s=1, c = color)
            ax.set_xlim([data_dict.min_pos[0], data_dict.max_pos[0]])
            ax.set_ylim([-data_dict.min_pos[2], -data_dict.max_pos[2]])
            ax.set_zlim([data_dict.min_pos[1], data_dict.max_pos[1]])
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')
            ax.set_facecolor('white')
            writer.add_figure('Surface', fig100, iter)  

            if args.point_cloud_out:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(surf_center_pos_np)
                pcd.normals = o3d.utility.Vector3dVector(surf_normals_np)
                o3d.io.write_point_cloud(str(pcd_dir / f"{iter}_pc.ply"), pcd)
                albedo_volume = self_supervised_optimizer.point_transient_renderer.get_albedo_np()
                np.save(str(pcd_dir / f"{iter}_iv"), volume)
                np.save(str(pcd_dir / f"{iter}_albedo_v"), albedo_volume)
                np.save(str(pcd_dir / f"{iter}_H_recon"), recon_tdata.detach().cpu().numpy())
                np.save(str(pcd_dir / f"{iter}_H"), ref_tdata.detach().cpu().numpy())
                np.save(str(pcd_dir / f"{iter}_kreal"), self_supervised_optimizer.phasor_field_imager.tkernel.real.detach().cpu().numpy())
                np.save(str(pcd_dir / f"{iter}_kimag"), self_supervised_optimizer.phasor_field_imager.tkernel.imag.detach().cpu().numpy())

            tkernel = self_supervised_optimizer.phasor_field_imager.get_tkernel()
            fig = plt.figure(dpi=300)
            tkernel_ylim = np.maximum(np.max(np.abs(tkernel.real)), np.max(np.abs(tkernel.imag)))
            plt.plot(tkernel.real)
            plt.plot(tkernel.imag)
            plt.ylim([-tkernel_ylim, tkernel_ylim])    
            writer.add_figure('kernel_t_domain', fig, iter)

            fig3 = plt.figure(dpi=600)            
            w = 4
            h = np.ceil(n_spots_per_relay_wall / w).astype(int)
            ylim = torch.max(torch.abs(origin_tdata.detach())).cpu().numpy()
            for i in range(h):
                for j in range(w):
                    global_index = j + i * w
                    if global_index < n_spots_per_relay_wall:
                        plt.subplot(h, w, global_index + 1)
                        title = f'({idx_h[global_index]}, {idx_w[global_index]})'
                        plt.title(title, fontsize=3, y=-0.2)
                        origin_tdata_view = origin_tdata[global_index, ...].detach().cpu().numpy()    # [T]
                        recon_tdata_view = recon_tdata[global_index, ...].detach().cpu().numpy()    # [T]
                        recon_tdata_no_ls_view = recon_tdata_no_ls[global_index, ...].detach().cpu().numpy()
                        plt.plot(origin_tdata_view, label='origin', linewidth=0.5)
                        plt.plot(recon_tdata_view, label='recon', linewidth=0.3)
                        plt.plot(begin_idx[idx_h[global_index], idx_w[global_index]].item(), 0, marker=".", markersize=2)
                        plt.xticks([])
                        plt.yticks([])
                        plt.ylim([-ylim/10, ylim])
            writer.add_figure('transients', fig3, iter)

            writer.flush()   

            print(f'End of iteration {iter:06d} / {args.max_iter - 1:06d}' \
                f' | total time: {str(datetime.timedelta(seconds=time.time()-total_start_time)).split(".")[0]}')
            print(f'[{args.name} @ GPU{args.rank}] loss = {loss:.6f}')

        # compute the backward gradients
        loss.backward()   
        torch.nn.utils.clip_grad_norm_(self_supervised_optimizer.parameters(), args.clip_grad_threshold)

        # update the parameters
        optimizer.step()
        self_supervised_optimizer.clip_parameters()

    # end of program
    if writer is not None:
        writer.close()        
