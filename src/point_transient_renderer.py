import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import (
    convolute_batch,
    convolute
)

###############################
# left-handed coordinate system
###############################
class PointTransientRenderer(nn.Module):
    def __init__(self, 
            args, 
            volume_albedo_init=None,
            threshold_intensity=0.0,
            device='cuda:0') -> None:
        super().__init__()
        self.hnum = args.hnum
        self.wnum = args.wnum
        self.tnum = args.tnum
        self.bin_len = args.bin_len
        self.device = device

        # same as phasor_field_imager
        x_aperture_size = np.maximum(args.laser_pos[...,0].max() -  args.laser_pos[...,0].min(), \
                            args.camera_pos[...,0].max() -  args.camera_pos[...,0].min())
        y_aperture_size = np.maximum(args.laser_pos[...,1].max() -  args.laser_pos[...,1].min(), \
                            args.camera_pos[...,1].max() -  args.camera_pos[...,1].min())
        self.aperture_full_size = np.array([y_aperture_size, x_aperture_size])        
        self.virtual_aperture_size = args.virtual_aperture_size
        self.min_pos = args.min_pos
        self.max_pos = args.max_pos
        self.threshold_intensity = threshold_intensity

        laserOrigin     = torch.from_numpy(args.laser_origin).to(device)
        cameraOrigin    = torch.from_numpy(args.camera_origin).to(device)
        laserPos        = torch.from_numpy(args.laser_pos).to(device)
        cameraPos       = torch.from_numpy(args.camera_pos).to(device)

        light_src_vec = laserOrigin - laserPos

        self.laser_pos = laserPos
        self.camera_pos = cameraPos
        self.light_src_vec_norm = F.normalize(light_src_vec, dim = -1) # [H, W, 3]
        self.d1 = torch.linalg.norm(laserPos - laserOrigin, dim = -1)    # [H, W]
        self.d4 = torch.linalg.norm(cameraPos - cameraOrigin, dim = -1)    # [H, W]
        self.wall_normal = torch.tensor([[0, 0, 1]]).to(device)    # [1, 3]

        default_sigma = args.laser_pulse_width * args.c_light / (2. * np.sqrt(2. * np.log(2))) / self.bin_len
        default_lamda = args.response_lambda
        default_dc = args.response_dc
        
        default_L_e = 1.0       # to be re-assigned in self_supervised_optimizer
        self.param_scale = nn.Parameter(torch.Tensor([default_L_e]))
        self.param_sigma = nn.Parameter(torch.Tensor([default_sigma]))
        self.param_lamda = nn.Parameter(torch.Tensor([default_lamda]))
        self.param_dc = nn.Parameter(torch.Tensor([default_dc]))

        if volume_albedo_init is not None:
            volume_mask = torch.zeros_like(volume_albedo_init)
            self.params_albedo = nn.Parameter(volume_mask)
        else:
            self.params_albedo = None
        self.phasor_volume = None
        self.laser_function = None
        self.sensor_function = None
        self.albedo_volume_tot = None
        self.albedo_interp_tot = None
        self.gt_sigma = self.param_sigma.data.to(self.device)
        self.gt_lamda = self.param_lamda.data.to(self.device)
        self.gt_dc = self.param_dc.data.to(self.device)


    def forward(self, surf_pos, surf_normals, xys, phasor_volume):
        # surf_pos: [MxN, 3]
        # surf_normals: [MxN, 3]
        # xys: [M, N, 2]
        y_idx = xys[:, 0, 0]    # [M,]
        x_idx = xys[:, 0, 1]    # [M,]
        self.phasor_volume = phasor_volume

        if self.params_albedo is not None:
            # volume_pos: [M, R, 3] ... (x, y, z)
            # nomalize the coordinates of volume_pos
            surf_pos_lh = surf_pos.clone()
            vs = surf_pos_lh.shape

            # volume_pos: [M, R, T, 3] ... (x, y, z)
            # nomalize the coordinates of volume_pos
            x_scale = (1.-(-1.))/self.virtual_aperture_size
            y_scale = (1.-(-1.))/self.virtual_aperture_size
            z_scale = (1.-(-1.))/(self.max_pos[2] - self.min_pos[2])
            z_offset = z_scale * (-self.min_pos[2]) - 1.
            surf_pos_lh[...,0] = surf_pos_lh[...,0] * x_scale
            surf_pos_lh[...,1] = -surf_pos_lh[...,1] * y_scale
            surf_pos_lh[...,2] = surf_pos_lh[...,2] * z_scale + z_offset            

            albedo_volume = self.get_albedo_volume()
            albedo_volume.clamp_(0.0, 1.0)    # albedo should be within 0.0~1.0
            volume_albedo = albedo_volume.permute(2, 0, 1)[None][None]  # [N=1, C=1, D, H, W]
            volume_grid = surf_pos_lh.reshape(-1, 3)[None][None][None]  # [N=1, d=1, h=1, w=MR, 3]
            volume_out = F.grid_sample(volume_albedo, volume_grid,
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True)    # [N=1, C=1, h=1, w=MR]
            
            albedo_interp_tot = volume_out.squeeze().reshape(*vs[:-1])
        else:
            albedo_volume_tmp = torch.ones_like(phasor_volume, dtype=torch.float32).to(self.device) 
            volume_mask = phasor_volume.detach() > self.threshold_intensity
            albedo_volume = albedo_volume_tmp * volume_mask            
            albedo_interp_tot = None

        self.albedo_interp_tot = albedo_interp_tot
        self.albedo_volume_tot = albedo_volume
        # tdata: [M, T]
        tdata_out, tdata = self.render_to_transient(surf_pos, surf_normals, y_idx, x_idx, albedo_interp_tot)

        return tdata_out, tdata

    def get_params_min_max(self):
        c_speed = 3e8
        N = 5   # hyper parameters of sinal ratio of 0bin to 1bin : 10^N
        tau_jitter_exp_max = 200e-12 * c_speed  # jitter maximum w.r.t. 10^(-N) of 0 bin
        lamda_constant = N * np.log(10.)
        lamda_min = self.bin_len / tau_jitter_exp_max * lamda_constant
        lamda_max = lamda_constant

        tau_pulse_min = 20e-12 * c_speed
        tau_pulse_max = 80e-12 * c_speed
        sigma_min = tau_pulse_min / self.bin_len / 2.355
        sigma_max = tau_pulse_max / self.bin_len / 2.355

        return sigma_min, sigma_max, lamda_min, lamda_max


    def get_diff_params(self):

        return (self.param_sigma.data - self.gt_sigma).detach().cpu().numpy(), \
            (self.param_lamda.data - self.gt_lamda).detach().cpu().numpy(), \
            (self.param_dc.data - self.gt_dc).detach().cpu().numpy()

 
    def get_albedo_np(self):
        volume = self.albedo_volume_tot.detach().cpu().numpy()
        
        return volume


    def get_albedo_volume(self):
        volume_mask = (self.phasor_volume.detach() > self.threshold_intensity) * 1.0
        albedo_volume = (self.params_albedo * volume_mask) + (self.phasor_volume * volume_mask)

        return albedo_volume
    

    def get_albedo_tv(self):
        # this is for backward AD
        if self.params_albedo is not None:
            albedo_volume = self.get_albedo_volume()
            volume_front, _ = albedo_volume.max(dim=-1)            

            nx, ny = volume_front.shape
            volume_grad_x = torch.cat([volume_front[1:,:] - volume_front[:-1,:],
                                    torch.zeros((1, ny), dtype=torch.float32).to(self.device)], dim=0)
            volume_grad_y = torch.cat([volume_front[:,1:] - volume_front[:,:-1],
                                    torch.zeros((nx, 1), dtype=torch.float32).to(self.device)], dim=1)

            # L2 is better than L1 due to the boundary is not clipped vertically or horizontally
            volume_tv_grad = torch.sqrt((volume_grad_x**2) + (volume_grad_y**2) + 1e-5)   # total variation L2 (isotropic)
            albedo_tv = volume_tv_grad.mean()            
        else:
            albedo_tv = 0.0
        
        return albedo_tv        

    def get_albedo_interp_np(self):
        if self.albedo_interp_tot is not None:
            res = self.albedo_interp_tot.cpu().detach().numpy()
        else:
            # dummy return
            res = np.zeros((3, 3))       

        return res
    

    def get_albedo_interp_count(self):
        if self.albedo_interp_tot is not None:
            res = ((self.albedo_interp_tot > self.threshold_intensity) * 1.0).mean()
        else:
            # dummy return
            res = torch.Tensor([0.0]).to(self.device)

        return res


    def generate_gaussian(self, sigma):
        knum = self.tnum // 4
        axis_t = (torch.arange(knum, dtype=torch.float32).to(self.device) - knum//2)
        sigma_inv = 1 / sigma        

        gauss_pulse = sigma_inv * (1./torch.sqrt(torch.Tensor([torch.pi * 2]).to(self.device))) * torch.exp(-(axis_t)**2 * (sigma_inv**2)/(2))
        gauss_pulse_norm = gauss_pulse

        return gauss_pulse_norm


    def generate_exponential(self, lamda):
        knum = self.tnum // 4
        axis_t = (torch.arange(knum, dtype=torch.float32).to(self.device) - knum//2)
        axis_t[axis_t < 0.0] = 0.0
        spad_response = (lamda * torch.exp(-lamda * axis_t)).clone()
        spad_response[:knum//2] = 0.      
        spad_response_norm = spad_response

        return spad_response_norm


    def generate_impulse_response(self, sigma, lamda):

        jitter_gauss = self.generate_gaussian(sigma)
        jitter_exp = self.generate_exponential(lamda)

        impulse_response = convolute(jitter_gauss, jitter_exp)

        return impulse_response
    

    def make_tmeas_to_realistic(self, tdata):
        impulse_response = self.generate_impulse_response(self.param_sigma, self.param_lamda)
        self.sensor_function = impulse_response
        
        tdata_conv = convolute_batch(tdata, impulse_response)  # sensor jitter model
        tdata_out = tdata_conv + self.param_dc    # ambient light + dark count rate

        return tdata_out


    def get_laser_function(self):

        if self.laser_function is not None:
            res = self.laser_function.detach().cpu().numpy()
        else:
            res = np.zeros(self.tnum // 4)
        return res


    def get_sensor_function(self):

        if self.sensor_function is not None:
            res = self.sensor_function.detach().cpu().numpy()
        else:
            res = np.zeros(self.tnum // 4)
        return res        


    def render_to_transient(self, surf_pos, surf_normals, y_idx, x_idx, albedo_interp=None):
       
        tnum = self.tnum

        light_vec = self.laser_pos[y_idx, x_idx, :].unsqueeze(1).expand_as(surf_pos) - surf_pos    # [M, N, 3]
        light_dist = torch.linalg.norm(light_vec, dim = -1) # [M, N]
        light_vec_norm = F.normalize(light_vec, dim = -1)             # [M, N, 3]

        spad_vec = self.camera_pos[y_idx, x_idx, :].unsqueeze(1).expand_as(surf_pos) - surf_pos    # [M, N, 3]
        spad_dist  = torch.linalg.norm(spad_vec, dim = -1)  # [M, N]
        spad_vec_norm = F.normalize(spad_vec, dim = -1)               # [M, N, 3]

        light_src_cosine = torch.sum(self.light_src_vec_norm[y_idx, x_idx, :].unsqueeze(1).expand_as(surf_pos) * self.wall_normal, dim=-1)

        # transient rendering for self-supervised learning
        light_intensity = self.param_scale      # constant illumination is assumed.
        dA = 1.                   # common differential of solid angle
        axis_t = torch.arange(tnum).to(self.device) * self.bin_len
        grid_t = torch.tile(axis_t, (surf_pos.shape[0], surf_pos.shape[1], 1))
        sigma_squared = self.bin_len ** 2 / (8. * np.log(2.))               # for Gaussian approx.
        if albedo_interp is None:
            albedo_interp = torch.ones((surf_pos.shape[0], surf_pos.shape[1]), dtype=torch.float32).to(self.device)

        # [M, N]
        value = light_intensity * albedo_interp / ((light_dist**2)*(spad_dist**2)) \
                * torch.sum(-light_vec_norm * self.wall_normal, dim=-1) \
                * torch.sum(-spad_vec_norm * self.wall_normal, dim=-1) \
                * torch.sum(light_vec_norm * surf_normals, dim=-1) \
                * torch.sum(spad_vec_norm * surf_normals, dim=-1) \
                * dA                        

        dist = self.d1[y_idx, x_idx].unsqueeze(1).expand_as(light_dist) + light_dist \
                + spad_dist + self.d4[y_idx, x_idx].unsqueeze(1).expand_as(light_dist)

        # differentiable version of transient generation (Gaussian approx.)
        dist_expanded = torch.tile(dist.unsqueeze(2), (1, 1, tnum))
        value_expanded = torch.tile(value.unsqueeze(2), (1, 1, tnum))
        
        tau = value_expanded * torch.exp((-1.) * (grid_t - dist_expanded)**2 / (2.*sigma_squared))
        tdata = torch.sum(tau, dim=1)

        tdata_out = self.make_tmeas_to_realistic(tdata)
        
        return tdata_out, tdata
