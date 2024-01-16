import torch
from utils import (
    RayBundle
)
import numpy as np
from random_spherical_samples import (
    concentric_mapping_square_to_hemisphere,
)
import torch.nn.functional as F


###############################
# left-handed coordinate system
###############################
class HemisphereRaysampler(torch.nn.Module):
    
    def __init__(self,
                args,
                n_rays_per_spot,
                n_spots_per_relay_wall, # Monte-Carlo number
                device = 'cuda:0'
    ) -> None:
        super().__init__()
        
        self.laser_pos = torch.from_numpy(args.laser_pos).float().to(device)
        self.camera_pos = torch.from_numpy(args.camera_pos).float().to(device)
        self.laser_origin = torch.from_numpy(args.laser_origin).float().to(device)
        self.camera_origin = torch.from_numpy(args.camera_origin).float().to(device)
        self.bin_len = args.bin_len
        self.n_pts_per_ray = args.tnum
        self.is_confocal = args.isConfocal
        self.hnum = args.hnum
        self.wnum = args.wnum
        self.n_spots_all = self.hnum * self.wnum
        self.n_rays_per_spot = n_rays_per_spot        
        self.n_spots_per_relay_wall = n_spots_per_relay_wall
        assert n_spots_per_relay_wall <= (self.n_spots_all), 'n_spots_per_relay_wall is too big!'
        self.idx_perm = torch.randperm(self.n_spots_all)

        self.device = device
    

    def forward(self) -> RayBundle:        
        # for exclusive sampling of spots within a single epoch
        hnum, wnum = self.hnum, self.wnum
        mc_size = self.n_spots_per_relay_wall       
        idx_perm = self.idx_perm[:mc_size]
        y_idx = torch.div(idx_perm, wnum, rounding_mode='trunc')
        x_idx = idx_perm % wnum        
        self.idx_perm = torch.roll(self.idx_perm, -mc_size)

        # for visualizing the progress of the center position's transients
        # for computing an initialization scale of the light intensity in renderer
        y_idx[0] = hnum // 2
        x_idx[0] = wnum // 2

        xys = torch.stack([y_idx, x_idx], dim=1).to(self.device).unsqueeze(1)   # [M, 1, 2]
        xys = xys.expand(xys.shape[0], self.n_rays_per_spot, xys.shape[2])              # [M, n_rays_per_spot, 2]

        d1 = torch.linalg.norm(self.laser_pos - self.laser_origin, dim = -1)      # [H, W]
        d4 = torch.linalg.norm(self.camera_pos - self.camera_origin, dim = -1)    # [H, W]
        d_offset = d1 + d4
        d1d4 = d_offset[y_idx, x_idx].unsqueeze(1).unsqueeze(2).expand(mc_size, self.n_rays_per_spot, self.n_pts_per_ray)    # [M, N, T]

        ####################################
        # origins
        ####################################
        origins = self.camera_pos[y_idx, x_idx, :]  # [M, 3]
        origins = origins.unsqueeze(1) # [M, 1, 3]
        origins = origins.expand(origins.shape[0], self.n_rays_per_spot, origins.shape[2]) # [M, n_rays_per_spot, 3]

        ####################################
        # directions
        ####################################
        fsphere_rand = concentric_mapping_square_to_hemisphere(int(np.sqrt(self.n_rays_per_spot)))    # [n_rays_per_spot, 3]
        fsphere_rand = fsphere_rand.reshape(-1, 3)

        directions = torch.from_numpy(fsphere_rand).float().to(self.device).unsqueeze(0)    # [1, n_rays_per_spot, 3]
        directions = directions.expand(mc_size, directions.shape[1], directions.shape[2])   # [M, n_rays_per_spot, 3]

        ####################################
        # lengths
        ####################################
        tau_ref = torch.arange(self.n_pts_per_ray).float().to(self.device) * self.bin_len
        tau_ref = tau_ref[None].expand(mc_size, self.n_pts_per_ray).clone().detach()    # for avoiding expand() memory issue.
        
        # compute the confocal case's values
        tau = tau_ref.unsqueeze(1).expand(mc_size, self.n_rays_per_spot, self.n_pts_per_ray)    # [M, N, T]
        d2d3 = tau - d1d4
        d2d3[d2d3 <= 0.0] = 1e-6
        t_conf = d2d3 / 2.
        t = t_conf
        
        # re-compute the non-confocal case's values
        if self.is_confocal == False:                        
            laser_pos_selected = self.laser_pos[y_idx, x_idx, :]    # [M, 3]
            laser_pos_selected = laser_pos_selected.unsqueeze(1).expand(mc_size, self.n_rays_per_spot, 3)

            l = torch.linalg.norm(laser_pos_selected-origins, dim=-1)
            lvec = F.normalize(laser_pos_selected-origins, dim=-1)
            dx = torch.sum(directions * lvec, dim=-1)
            dy = (1.0 - dx**2)**0.5

            # this is just for processing an exceptional case
            idx_offset = torch.ceil((torch.linalg.norm((self.laser_pos - self.camera_pos), dim=-1) - d_offset) / self.bin_len).to(self.device)
            offset = idx_offset[y_idx, x_idx].long()
            for i in range(offset.shape[0]):
                tmp = tau_ref[i,offset[i]]
                tau_ref[i,:offset[i]] = tmp

            # all vectors are converted to the form of [M, n_rays_per_pixel, n_pts_per_ray]
            l = l.unsqueeze(2).expand(mc_size, self.n_rays_per_spot, self.n_pts_per_ray)
            dx = dx.unsqueeze(2).expand(mc_size, self.n_rays_per_spot, self.n_pts_per_ray)
            dy = dy.unsqueeze(2).expand(mc_size, self.n_rays_per_spot, self.n_pts_per_ray)
            tau = tau_ref.unsqueeze(1).expand(mc_size, self.n_rays_per_spot, self.n_pts_per_ray)
            t_nonconf = (dx * l + tau) * ((tau**2) - (l**2)) / (2 * ((dx**2) * ((tau**2) - (l**2)) + (dy**2) * (tau**2)))
            
            for i in range(mc_size):
                if t_nonconf[i, ...].isnan().any():
                    t_nonconf[i, ...] = t_conf[i, ...]            
            t = t_nonconf
        
        ray_bundle = RayBundle(origins=origins, directions=directions, lengths=t, xys=xys)

        return ray_bundle
