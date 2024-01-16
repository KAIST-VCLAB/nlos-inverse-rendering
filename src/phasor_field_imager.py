import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import replace_denormals

# This code is mostly derived from the paper below.
# @article{liu2020phasor,
#   title={Phasor field diffraction based reconstruction for fast non-line-of-sight imaging systems},
#   author={Liu, Xiaochun and Bauer, Sebastian and Velten, Andreas},
#   journal={Nature communications},
#   volume={11},
#   number={1},
#   pages={1645},
#   year={2020},
#   publisher={Nature Publishing Group UK London}
# }


class PhasorFieldImager(nn.Module):
    def __init__(self, 
                args, 
                init_with_original_kernel=False,
                init_scale_sigma=1.2,
                init_scale_omega=0.8,
                freq_band_enable = False,                
                distance_factor=2,
                threshold=0.0,
                depth_mode=0,
                device='cuda:0'):
        super().__init__()
        
        self.device = device
        self.bin_len = args.bin_len
        self.sampling_grid_spacing = args.sampling_grid_spacing    
        self.min_pos = args.min_pos
        self.max_pos = args.max_pos
        self.lambda_times = args.lambda_times
        self.freq_band_enable = freq_band_enable
        self.init_good = init_with_original_kernel
        self.tnum = args.tnum
        self.cycle_times = args.cycle_times
        x_aperture_size = np.maximum(args.laser_pos[...,0].max() -  args.laser_pos[...,0].min(), \
                            args.camera_pos[...,0].max() -  args.camera_pos[...,0].min())
        y_aperture_size = np.maximum(args.laser_pos[...,1].max() -  args.laser_pos[...,1].min(), \
                            args.camera_pos[...,1].max() -  args.camera_pos[...,1].min())
        self.peak_ratio = 1e-1
        self.aperture_full_size = np.array([y_aperture_size, x_aperture_size])
        self.z_gate = 0     # for removing the direct reflection of the relay wall
        self.d_offset = 0.0
        self.c_light = 299792458    # speed of light [m/s]
        self.confocal = args.isConfocal     # confocal sampling
        self.tkernel_length = 0

        tag_maxdepth = self.max_pos[-1]     # target max depth (m)
        nMax = self.col_round(2 * tag_maxdepth / self.bin_len)   # max time bin count (assume: round-trip time)
        self.nMax = nMax
        self.virtual_aperture_size = args.virtual_aperture_size
        self.kernel_dim = 2*self.col_round(self.col_round(self.virtual_aperture_size/(self.sampling_grid_spacing))/2)
        wavelength = args.lambda_times * 2 * args.sampling_grid_spacing
        self.default_omega = 2. * torch.pi / wavelength
        self.default_sigma = self.cycle_times * wavelength / 6
        
        kernel_sigma_inv = 1.0 / self.default_sigma
        kernel_omega = self.default_omega
        kernel_params = torch.Tensor([kernel_sigma_inv, kernel_omega]).to(self.device)
        grids_k = torch.arange(self.nMax, dtype=torch.float32) - round(self.nMax/2)
        self.grids_k = torch.fft.ifftshift(grids_k.to(self.device) * self.bin_len)
        
        # searching initial frequency_index and marking it fixed
        weight, _, frequency_index, lambda_loop, omega_space = self.generate_fourierD_kernel(kernel_params, init=True)
        self.fnum = weight.shape[0]
        self.frequency_index = frequency_index
        self.lambda_loop = lambda_loop
        self.omega_space = omega_space

        if not self.init_good:
            sigma_inv = 1.0 / (self.default_sigma * init_scale_sigma * torch.ones_like(kernel_params[...,0]).to(self.device))
            omega = 2. * torch.pi / (wavelength * init_scale_omega * torch.ones_like(kernel_params[...,1]).to(self.device))
            kernel_params[0] = sigma_inv
            kernel_params[1] = omega

        self.params = nn.Parameter(kernel_params)

        self.phasor_volume = None
        self.tkernel = None
        self.fkernel_mag = None
        self.fkernel_phase = None
        self.distance_factor = distance_factor
        self.threshold = threshold      # this is for debugging
        self.phasor_volume_RSD = None
        self.depth_mode = depth_mode


    def forward(self, x, volume_pos=None):
        u_total = self.generate_fourierD_hist(x)
        phasor_volume = self.main_focusing(u_total)

        vs = volume_pos.shape
        volume_pos_lh = volume_pos.clone()

        # volume_pos: [M, R, T, 3] ... (x, y, z)
        # nomalize the coordinates of volume_pos
        # Caution: aperture center should be (0,0)
        x_scale = (1.-(-1.))/self.virtual_aperture_size
        y_scale = (1.-(-1.))/self.virtual_aperture_size
        z_scale = (1.-(-1.))/(self.max_pos[2] - self.min_pos[2])
        z_offset = z_scale * (-self.min_pos[2]) - 1.
        volume_pos_lh[...,0] = volume_pos[...,0] * x_scale
        volume_pos_lh[...,1] = -volume_pos[...,1] * y_scale    # just for following grid_sample axis convention
        volume_pos_lh[...,2] = volume_pos[...,2] * z_scale + z_offset

        self.phasor_volume = phasor_volume

        phasor_volume = phasor_volume.permute(2, 0, 1)[None][None]  # [N=1, C=1, D, H, W]
        volume_grid = volume_pos_lh.reshape(-1, 3)[None][None][None]  # [N=1, d=1, h=1, w=MRT, 3]
        volume_out = F.grid_sample(phasor_volume, volume_grid,
                                mode='bilinear', padding_mode='zeros',
                                align_corners=True)    # [N=1, C=1, h=1, w=MRT]
        
        intensity_out = volume_out.squeeze().reshape(*vs[:-1])       

        return intensity_out, self.phasor_volume


    def get_volume_image(self):
        if self.phasor_volume is not None:
            volume = self.phasor_volume.detach().cpu().numpy()
            img = np.max(volume, axis=-1)
        else:
            h, w, _ = self.phasor_volume.shape
            img = np.zeros((h, w))
        
        return img

    def get_volume(self):

        return self.phasor_volume.detach().clone()

    def get_volume_np(self):

        return self.phasor_volume_RSD.detach().cpu().numpy()


    def get_kernel(self):
        # this is for visualizing

        return self.fkernel_mag.detach().cpu().numpy(), self.fkernel_phase.detach().cpu().numpy()


    def get_tkernel(self):
        # this is for visualizing
        tkernel = self.tkernel.detach().cpu().numpy()
        sigma = 1 / (self.params.data[0].detach().cpu().numpy().min() + 1e-5)
        sigma = np.abs(sigma)
        bin_count = int(np.round(4 * sigma / self.bin_len))
        if bin_count > (self.nMax//2):
            bin_count = (self.nMax//2)
        tkernel_cat = np.concatenate([tkernel[..., -bin_count:], tkernel[..., :bin_count]], axis=-1)

        return tkernel_cat

    
    def get_volume_tv(self):
        # this is for backward AD
        phasor_volume = self.phasor_volume

        volume_max_bottom, _ = phasor_volume.max(dim=1)
        volume_tv = volume_max_bottom.mean()
        
        return volume_tv

    def col_round(self, x):
        import math
        frac = x - math.floor(x)
        if frac < 0.5: return math.floor(x)
        return math.ceil(x)
        

    def create_virtual_aperture(self, uin, apt_in, maxSz, posL):

        def tool_symmetry(u1, phyinapt):

            M, N = u1.shape
            delta = phyinapt[0]/(M-1)

            if M > N:
                square_pad = np.ceil(0.5 * (M - N)).astype(int)
                u2 = F.pad(u1, (square_pad, square_pad, 0, 0), 'constant', 0)
                phyoutapt = delta * u2.shape[0]
            elif M < N:
                square_pad = np.ceil(0.5 * (N - M)).astype(int)
                u2 = F.pad(u1, (0, 0, square_pad, square_pad), 'constant', 0)
                phyoutapt = delta * u2.shape[0]
            else:
                u2 = u1
                phyoutapt = phyinapt

            return u2, phyoutapt


        # % Get input wavefront dimension
        M, N = uin.shape
        
        # % Calculate the spatial sampling density, unit meter
        delta = apt_in[0]/(M-1)
        
        # % Calculate the ouptut size
        M_prime = self.col_round(maxSz/delta)
        
        # % symmetry square padding
        if M == N:
            pass
        else:
            uin, _ = tool_symmetry(uin, apt_in)
        
        # % Padding size difference
        uin_shape = np.array([*uin.shape])
        diff = uin_shape - np.array([M, N])
        posL = posL + diff/2    # update the virtual point source location after symmetry
        
        # % Update the symmetry aperture size
        M, _ = uin.shape
        
        # % Virtual aperture extented padding on the boundary
        # % difference round to nearest even number easy symmetry padding
        dM = 2 * np.ceil((M_prime - M)/2).astype(int)
        
        # % symmetry padding on the boundary
        if dM > 0:
            # % Using 0 boundary condition
            dM_2 = int(dM/2)
            uout = F.pad(uin, (dM_2, dM_2, dM_2, dM_2), 'constant', 0)
            posO = posL + dM_2     # update the virtual point source location after padding
        else:
            uout = uin
            posO = posL
        
        # % update the virtual aperture size
        apt_out = delta * np.array([*uout.shape])

        return uout, apt_out, posO


    def camera_focusing(self, uin, L, wavelength, depth, alpha):
        # inner functions in camera_focusing()
        def tool_field_zeropadding(u1, phyinapt, wavelength, depth, alpha):
            M, N = u1.shape
            delta = phyinapt(1)/(M-1)
            
            if M == N:
                pass
            else:
                u1, _ = tool_symmetry(u1, phyinapt)
            
            sM, _ = u1.shape
            
            N_uncertaint = (wavelength * abs(depth))/(delta**2)
            pad_size = (N_uncertaint - sM)/2

            if pad_size > 0:
                pad_size = self.col_round(alpha * pad_size)
            else:
                pad_size = 0
            u2 = F.pad(u1, (pad_size, pad_size, pad_size, pad_size), 'constant', 0)
            phyoutapt = delta * (u2.shape[0]-1)
            
            # inner functions in tool_field_zeropadding()
            def tool_symmetry(u1, phyinapt):

                M, N = u1.shape
                delta = phyinapt[0]/(M-1)

                if M > N:
                    square_pad = np.ceil(0.5 * (M - N)).astype(int)
                    u2 = F.pad(u1, (0, 0, square_pad, square_pad), 'constant', 0)
                    phyoutapt = delta * u2.shape[0]
                elif M < N:
                    square_pad = np.ceil(0.5 * (N - M)).astype(int)
                    u2 = F.pad(u1, (square_pad, square_pad, 0, 0), 'constant', 0)
                    phyoutapt = delta * u2.shape[0]
                else:
                    u2 = u1
                    phyoutapt = phyinapt

                return u2, phyoutapt
            
            return u2, phyoutapt, pad_size, sM

        # inner functions in camera_focusing()
        def propRSD_conv(u1, L, wavelength, z):
            M, N = u1.shape			# get input field array size, physical size

            # % Extract each physical dimension
            l_1 = L[0]
            l_2 = L[1]
            
            # % Spatial sampling interval
            dx = l_1/(M-1)						# sample interval x direction
            dy = l_2/(N-1)						# sample interval y direction

            # % spatial sampling resolution needs to be equal in both dimension
            z_hat = z/dx;                   
            mul_square = wavelength * z_hat/(M * dx)

            # % center the grid coordinate
            m = torch.linspace(0, M-1, M).to(self.device)
            m = m - M/2
            n = torch.linspace(0, N-1, N).to(self.device)
            n = n - N/2
            
            g_m, g_n = torch.meshgrid(n, m)  # coordinate mesh			
            g_m = g_m.to(self.device)
            g_n = g_n.to(self.device)

            if self.confocal:
                coeff = 2.
            else:
                coeff = 1.

            # %  Convolution Kernel Equation including the firld drop off term
            h = torch.exp(
                    1j * coeff * 2 * torch.pi \
                    * ((z_hat**2) * torch.sqrt(1 + (g_m**2)/(z_hat**2) + (g_n**2)/(z_hat**2)) / (mul_square * M)) \
                    ) \
                    / torch.sqrt(1 + (g_m**2)/(z_hat**2) + (g_n**2)/(z_hat**2))
            
            # % Convolution or multiplication in Fourier domain
            H = torch.fft.fft2(h)
            U1 = torch.fft.fft2(u1)
            U2 = U1 * H
            u2 = torch.fft.ifftshift(torch.fft.ifft2(U2))   # omitting ifftshift for test is necessary
                        
            return u2        

        # implementation of camera_focusing()
        if alpha != 0:
            u1_prime, aperturefullsize_prime, pad_size, Nd = tool_field_zeropadding(uin, L, wavelength, depth, alpha)
        else:
            u1_prime = uin
            aperturefullsize_prime = L
            pad_size = 0
            Nd = u1_prime.shape[0]
        
        uout = torch.fliplr(propRSD_conv(u1_prime, aperturefullsize_prime, wavelength, depth))
        uout = uout[pad_size:pad_size+Nd, pad_size: pad_size+Nd]

        return uout


    def main_focusing(self, u_total):
        sample_spacing = self.sampling_grid_spacing
        aperturefullsize = self.aperture_full_size
        # virtual aperture size : squared grid area width (=height)
        v_apt_Sz = self.virtual_aperture_size

        # force to apply parameter constraints (abs = self.default_weight)
        weight_mag, weight_phase, _, _, _ = self.generate_fourierD_kernel(self.params, init=False)
        lambda_loop = self.lambda_loop
        omega_space = self.omega_space
        weight = weight_mag * torch.exp(1j * weight_phase)

        # Pad Virtual Aperture Wavefront
        # additional if physical dimension is odd number
        if u_total.shape[0] % 2 == 1:
            tmp_x, tmp_y, tmp_z = u_total.shape
            tmp3D = torch.zeros((self.col_round(tmp_x/2)*2, self.col_round(tmp_y/2)*2, tmp_z), dtype=torch.complex64).to(self.device) # round --> col_round
            
            aperturefullsize = np.array([tmp3D.shape[0]-1, tmp3D.shape[1]-1]) * sample_spacing
            tmp3D[:u_total.shape[0], :u_total.shape[1], :] = u_total
            u_total = tmp3D

        # u_tmp = torch.zeros_like(u_total).to(self.device)
        # perallocated memeory for padding wavefront
        u_tmp = torch.zeros((2 * self.col_round(self.col_round(v_apt_Sz/(sample_spacing))/2), \
                            2 * self.col_round(self.col_round(v_apt_Sz/(sample_spacing))/2), \
                            u_total.shape[2]), dtype=torch.complex64).to(self.device)        

        # create Virtual Aperture by zero padding
        for index in range(u_total.shape[2]):
            tmp = u_total[:,:,index]
            u_tmp[:,:,index], apt_tmp, _ = self.create_virtual_aperture(tmp, aperturefullsize, v_apt_Sz, 0)

        aperturefullsize = apt_tmp      # update virtual aperture size
        u_total = u_tmp     # update wavefront cube

        # create depth slice for the volume
        depth_min = self.min_pos[2]
        depth_max = self.max_pos[2]

        if self.depth_mode == 0:
            depth_loop = torch.arange(depth_min, depth_max, 2.0 * sample_spacing).to(self.device)            
        elif self.depth_mode == 1:
            depth_loop = torch.arange(depth_min, depth_max, 2.0 * sample_spacing / 2).to(self.device)
        elif self.depth_mode == 2:
            depth_loop = torch.arange(depth_min, depth_max, self.bin_len).to(self.device)

        # Reconstruction using fast RSD
        nZ = depth_loop.shape[0]
        u_volume = torch.zeros((u_total.shape[0], u_total.shape[1], nZ), dtype=torch.complex64).to(self.device)

        for i in range(nZ):
            depth = depth_loop[i]
            u_tmp = torch.zeros((u_total.shape[0], u_total.shape[1]), dtype=torch.complex64).to(self.device) 

            if self.confocal:
                illum_shift = 0.
            else:
                illum_shift = depth                

            for spectrum_index in range(lambda_loop.shape[0]):
                u_field = u_total[:,:,spectrum_index]
                wavelength = lambda_loop[spectrum_index]
                omega = omega_space[spectrum_index]
                weight_for_apply = weight[spectrum_index]

                u1 = u_field * weight_for_apply * torch.exp(1j * omega * (illum_shift + self.d_offset)/self.c_light \
                                        * torch.ones(u_field.shape, dtype=torch.float32).to(self.device))   # integral form of ifft
                u2_RSD_conv = self.camera_focusing(u1, aperturefullsize, wavelength, depth, 0)
                u_tmp = u_tmp + u2_RSD_conv

            u_volume[:,:,i] = u_tmp

        mgn_volume = torch.abs(u_volume)
        out_volume = torch.flip(mgn_volume, [1])
        out_volume[..., -3:] = 0.0  # remove fft artifact

        out_volume_vis = out_volume.detach()
        out_volume_vis_max = out_volume_vis.max()
        self.phasor_volume_RSD = out_volume_vis / out_volume_vis_max

        h, w, d = out_volume.shape
        dist = torch.linspace(self.min_pos[2], self.max_pos[2], d).to(self.device)
        dist_volume = torch.tile(dist**self.distance_factor, (h, w, 1))
        
        albedo_volume_raw = out_volume * dist_volume
        albedo_volume_max = albedo_volume_raw.max()
        res_out_volume = albedo_volume_raw / albedo_volume_max
        assert res_out_volume.isnan().any()==False, f'albedo_volume_max seems to be zero'
        
        if self.threshold > 0.0:
            res_out_volume[res_out_volume < self.threshold] = 0.0

        return res_out_volume


    # Fourier kernel generation
    def generate_fourierD_kernel(self, kernel_params, init=True):
        ts = self.bin_len / self.c_light    # sampling time period (sec)
        nMax = self.nMax

        sigma_inv = kernel_params[0]
        omega = kernel_params[1]
        # mu = kernel_params[2]

        t = self.grids_k
        cmp_sin = torch.exp(1j * omega * t)
        gauss_wave = torch.exp(-(t * t * sigma_inv * sigma_inv) / 2)
        tkernel = cmp_sin * gauss_wave        
        tkernel = tkernel.type(torch.complex64)
        self.tkernel = tkernel
        assert tkernel.abs().max() > 0, f'tkernel seems to be zeros'


        peak_ratio = self.peak_ratio
        fkernel = torch.fft.fft(tkernel)/nMax
        P_wave_ = torch.abs(fkernel)        
        fkernel_real = replace_denormals(fkernel.real)  # just for escaping NaN
        P_wave_phase_ = torch.atan2(fkernel.imag,  fkernel_real)   # just for escaping NaN

        P_wave = P_wave_[..., :self.col_round(nMax/2+1)]        
        P_wave_phase = P_wave_phase_[..., :self.col_round(nMax/2+1)]

        self.fkernel_mag = P_wave
        self.fkernel_phase = P_wave_phase

        # Find  the filtered frequency index (only index, no unit associate)
        if init:
            band_max = self.col_round(nMax/2+1)
            coeff_ratio = P_wave/torch.max(P_wave)
            frequency_index = torch.argwhere(coeff_ratio>=peak_ratio).squeeze()                
            if self.freq_band_enable:
                band_start = frequency_index[0] - 5
                band_end = frequency_index[-1] + 10
                if band_start < 0:
                    band_start = 0
                if band_end >= band_max:
                    band_end = band_max - 1
                frequency_index = torch.arange(band_start, band_end+1).to(self.device)
            else:
                # Calculate the ratio based on the Energy peak value
                band_start = frequency_index[0]
                band_end = frequency_index[-1]
    
            # Calculate the final synethsis quantity
            fs = 1/ts
            fre_mask = fs * (frequency_index+1) / nMax
            omega_space = 2 * torch.pi * fre_mask
            lambda_loop = self.c_light * 2 * torch.pi / omega_space
            weight = P_wave[frequency_index]     # weight from the gaussian spectrum
            weight_phase = P_wave_phase[frequency_index]
        else:
            frequency_index = self.frequency_index
            weight = P_wave[..., frequency_index]     # weight from the gaussian spectrum
            weight_phase = P_wave_phase[..., frequency_index]
            lambda_loop = None
            omega_space = None

        return weight, weight_phase, frequency_index, lambda_loop, omega_space


    # Fourier histogram generation
    def generate_fourierD_hist(self, x):
        rect_data = x
        M, N, T = rect_data.shape
        rect_data[..., :self.z_gate] = 0.   # filtering out the direct reflection
        if T > self.nMax:
            rect_data = rect_data[...,:self.nMax]    # filter out the unnecessary parts        

        frequency_index = self.frequency_index
        omega_space = self.omega_space

        num_component = omega_space.shape[0]
        u_total = torch.zeros((M, N, num_component), dtype=torch.complex64).to(self.device)
        for ii in range(M):
            for jj in range(N):
                t_tmp = rect_data[ii, jj, :]
                f_tmp = torch.fft.fft(t_tmp)
                f_slice = f_tmp[frequency_index.squeeze()]
                u_total[ii,jj,:] = f_slice.squeeze()

        return u_total


    # just for default imaging
    def generate_phasor_volume(self, x):
        with torch.no_grad():
            u_total = self.generate_fourierD_hist(x)
            out_volume = self.main_focusing(u_total)

        return out_volume.detach().cpu().numpy()        
    
    def generate_phasor_volume_optim(self, x):
        u_total = self.generate_fourierD_hist(x)
        out_volume = self.main_focusing(u_total)

        return out_volume
    