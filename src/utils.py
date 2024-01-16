import torch
import torch.nn.functional as F
from typing import NamedTuple


class RayBundle(NamedTuple):

    origins: torch.Tensor
    directions: torch.Tensor
    lengths: torch.Tensor
    xys: torch.Tensor


def ray_bundle_to_ray_points(ray_bundle: RayBundle) -> torch.Tensor:

    return ray_bundle_variables_to_ray_points(
        ray_bundle.origins, ray_bundle.directions, ray_bundle.lengths
    )


def ray_bundle_variables_to_ray_points(
    rays_origins: torch.Tensor,
    rays_directions: torch.Tensor,
    rays_lengths: torch.Tensor,
) -> torch.Tensor:

    rays_points = (
        rays_origins[..., None, :]
        + rays_lengths[..., :, None] * rays_directions[..., None, :]
    )
    return rays_points


# extracts the length of max value
def softargmax3(
    volume:torch.tensor,    # [M, N, T]
    lengths:torch.tensor,      # [M, N, T]
    beta:float = 1000., 
    eps:float = 1e-5
    ):
    # beta: hyper parameter for softmax()
    # eps: threshold for valid intensity

    # soft arg max routine
    volume_indices = lengths

    beta_volume = beta*volume   # amplified probability for each voxel
    beta_volume_max, _ = torch.max(beta_volume, dim=-1)
    beta_volume_norm = beta_volume - beta_volume_max.unsqueeze(-1).expand(beta_volume.shape)    # normalize
    volume_softmax = F.softmax(beta_volume_norm, dim=-1)

    volume_product = volume_indices * volume_softmax # in order to extract approx. max value
    depth_map = torch.sum(volume_product, dim=-1)

    # exceptional case: what if all the values are zeros in z-direction?
    # then the depth value converges to max_depth which is assigned as lengths[0,0,-1]
    volume_max, _ = torch.max(volume, dim=-1)
    depth_map[volume_max < eps] = lengths[0,0,-1]

    return depth_map


def get_tmeas_scale(tmeas:torch.Tensor, clip_percent=0.999999):
    N = 1000
    tmeas_hist, _ = torch.histogram(tmeas.reshape(-1).cpu(), N)
    tmeas_cdf = torch.cumsum(tmeas_hist, dim=-1)

    threshold = tmeas_cdf[-1] * clip_percent
    threshold_mask = tmeas_cdf > threshold
    indices = threshold_mask.nonzero()  
    tmeas_scale = indices[0][0]/N * tmeas.max()

    return tmeas_scale


def get_figure(arr, lim='auto', cmap='turbo'):
    import matplotlib.pyplot as plt
    assert len(arr.shape) == 2
    if isinstance(lim, str) and lim == 'auto':
        lower, upper = arr.min(), arr.max()
    elif isinstance(lim, tuple):
        lower, upper = lim
    im_ratio = arr.shape[0]/arr.shape[1]
    s = 6 # width
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(s*1.2,im_ratio*s))
    im = ax.imshow(arr, cmap=cmap, clim=[lower, upper])
    fig.colorbar(im, ax=ax, fraction=0.047*im_ratio, pad=0.04)
    fig.tight_layout()

    return fig


def get_figures(arr, lim='auto', cmap='turbo'):
    import matplotlib.pyplot as plt
    assert len(arr.shape) == 3

    if isinstance(lim, str) and lim == 'auto':
        lower, upper = arr.min(), arr.max()
    elif isinstance(lim, tuple):
        lower, upper = lim
   
    title = ['Front', 'Side', "Bottom"]
    row = [0, 0, 1]
    col = [0, 1, 0]    

    fig, ax = plt.subplots(nrows=2, ncols=2, dpi=300)
    for i in range(3):
        arr_=arr.max(axis=2-i)
        if i==2:
            arr_ = arr_.transpose()
            fig.colorbar(im, ax=ax[r,c], pad=0.04)
        r = row[i]
        c = col[i]
        # arr_ratio = arr_.shape[0] / arr_.shape[1]
        im = ax[r, c].imshow(arr_, cmap=cmap, clim=[lower, upper])
        ax[r, c].set_title(title[i])
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].set
        # fig.colorbar(im, ax=ax[i], fraction=0.047*arr_ratio, pad=0.04)
    ax[1, 1].axis('off')
    fig.tight_layout()

    return fig


def debug_to_file(commit=True, **kwargs):
    if commit:
        import sys
        print('usage: debug_to_file(True, a=a, b=b)')
        db = {}
        for key, value in kwargs.items():
            db[key] = value
            print(key)        
        torch.save(db, '../../../data/debug_db')
        print('Debug file exported!')
        sys.exit(0)


def convolute(input, kernel, device='cuda:0'):
    K_half = kernel.shape[0] // 2
    K_sign = kernel.shape[0] % 2
    if K_sign == 0:
        input_padded = F.pad(input, (K_half,K_half-1), 'constant', 0)
    else:
        input_padded = F.pad(input, (K_half,K_half), 'constant', 0)
    kernel_flip = torch.flip(kernel, [0])
    output = F.conv1d(input_padded[None][None], kernel_flip[None][None], padding='valid')

    return output[0, 0, :]


def convolute_batch(input, kernel):
    # input: [M, T]
    # kernel: [T]
    # output: [M, T]
    K_half = kernel.shape[0] // 2
    K_sign = kernel.shape[0] % 2
    if K_sign == 0:
        input_padded = F.pad(input, (K_half,K_half-1), 'constant', 0)
    else:
        input_padded = F.pad(input, (K_half,K_half), 'constant', 0)
    kernel_flip = torch.flip(kernel, [0])
    input_padded = input_padded.unsqueeze(1)
    output = F.conv1d(input_padded, kernel_flip[None][None], padding='valid')
    output = output.squeeze()

    return output    


def penalize_min_max(input:torch.Tensor, min_value:float, max_value:float):
    if input is not None:
        loss_term = ((input - min_value) ** 2) * (input < min_value) + ((input - max_value) ** 2) * (input > max_value)
        res = loss_term.mean()
    else:
        res = 0.0    
    return res


def replace_denormals(x: torch.Tensor, threshold=1e-10):
    y = x.clone()
    y[(x < threshold) & (x > -1.0 * threshold)] = threshold
    return y

