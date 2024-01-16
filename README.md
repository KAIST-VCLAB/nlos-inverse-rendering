# Self-Calibrating, Fully Differentiable NLOS Inverse Rendering

Existing time-resolved non-line-of-sight (NLOS) imaging methods reconstruct hidden scenes by inverting the optical paths of indirect illumination measured at visible relay surfaces. These methods are prone to reconstruction artifacts due to inversion ambiguities and capture noise, which are typically mitigated through the manual selection of filtering functions and parameters. We introduce a fully-differentiable end-to-end NLOS inverse rendering pipeline that self-calibrates the imaging parameters during the reconstruction of hidden scenes, using as input only the measured illumination while working both in the time and frequency domains. Our pipeline extracts a geometric representation of the hidden scene from NLOS volumetric intensities and estimates the time-resolved illumination at the relay wall produced by such geometric information using differentiable transient rendering. We then use gradient descent to optimize imaging parameters by minimizing the error between our simulated time-resolved illumination and the measured illumination. Our end-to-end differentiable pipeline couples diffraction-based volumetric NLOS reconstruction with path-space
light transport and a simple ray marching technique to extract detailed, dense sets of surface points and normals of hidden scenes. Our results demonstrate the robustness of our method to consistently
reconstruct geometry and albedo, even with significant noise interference.

### [Project page](https://vclab.kaist.ac.kr/siggraphasia2023/index.html) | [Paper](https://vclab.kaist.ac.kr/siggraphasia2023/nlospaper23-7.pdf) | [Supplemental](https://vclab.kaist.ac.kr/siggraphasia2023/nlospaper23-7-supple.pdf)
[Kiseok Choi](http://vclab.kaist.ac.kr/kschoi/index.html), 
[Inchul Kim](https://inchul-kim.github.io/), 
[Dongyoung Choi](http://vclab.kaist.ac.kr/dychoi/index.html), 
[Julio Marco](http://webdiis.unizar.es/~juliom/), 
[Diego Gutierrez](http://giga.cps.unizar.es/~diegog/), 
[Min H. Kim](http://vclab.kaist.ac.kr/minhkim/index.html)

## Setup
Make sure that your hardware can run PyTorch 1.12.1. We provide 2 options. We highly recommend option 1 (Docker with CUDA GPU support) to prevent conflict with your local environment.
### Option 1. Docker + CUDA
Docker CUDA support ([NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)): Available for Linux / Windows 11 WSL. In case of Windows 10 WSL, you need 21H2 update [[Linux reference](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), [Windows reference](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)].
Make sure you properly installed Docker CUDA support. The following command builds a docker image with ./Dockerfile. This may take some time.
```
docker build -t nlos:v0 .
```
Start the docker container. You may modify some options (e.g. --name for container name (2nd line), --gpu for gpu index (3rd line), additional -v option for mount volume (4th line)).
```
docker run -ti \
--name nlos \
--gpus all \
-v "$CODE_PATH":/workspace/code \
-v "$DATA_PATH":/workspace/data \
nlos:v0
```

### Option 2. Non-Docker + CUDA
Install the Anaconda environment ([reference](https://docs.anaconda.com/free/anaconda/install/linux/)).
Create a conda environment and activate it.
```
conda create -n nlos python=3.7
conda activate nlos
```
Install the related packages using the following commands.
```
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install opencv-python-headless opencv-contrib-python-headless tensorboard
pip install -U matplotlib
conda install -c anaconda h5py
conda install -c conda-forge easydict
pip install -U open3d
pip install mat73
```

## Tested Hardware Environment
```
CPU: AMD EPYC 7763
GPU: NVIDIA A100 80GB
RAM: 1TB
```

## Data Preparation
Download the pre-processed scene [bike](https://drive.google.com/file/d/1o3Sj1rs5frB2QVpuJ4-W5-WBE-h2Stm6/view?usp=drive_link).
This scene has been captured from Lindell et al.(https://www.computationalimaging.org/publications/nlos-fk/) and has been pre-processed using their MATLAB code for reducing the resolution to 256 x 256 x 512.
Once the downloading is completed, move the scene file to a specific directory and modify the data path in "./config/default_scenario.json" accordingly.

## Optimization
Run the optimization program using the following command. The default value of "$JSON_FILE_PATH" is "./config/default_scenario.json".
```
python src/main.py --config_path "$JSON_FILE_PATH"
```
Once the optimization process is completed, with 170 iterations (it took about 1.73 hours on the tested hardware), the resulting point cloud and surface normals will be saved in the directory "./results_${JSON_FILE_NAME}/pcd" under the filename "${ITERATION_NUMBER}_pc.ply." Additionally, the volumetric intensity data will also be stored with the name "${ITERATION_NUMBER}_iv.npy," and the albedo information will be saved as "${ITERATION_NUMBER}_albedo_v.npy" within the same directory.

## Citation
```
@InProceedings{Choi:SIGGRAPHAsia:2023,
author  = {Kiseok Choi and Inchul Kim and Dongyoung Choi and Julio Marco 
           and Diego Gutierrez and Min H. Kim},
title   = {Self-Calibrating, Fully Differentiable NLOS Inverse Rendering},
booktitle = {Proceedings of ACM SIGGRAPH Asia 2023},
month   = {December},
year    = {2023},
}
```

## Acknowledgements
We employed the [Fast Phasor-field NLOS](https://biostat.wisc.edu/~compoptics/phasornlos20/fastnlos.html) code as the primary imaging algorithm and integrated certain code structures from [PyTorch3D](https://pytorch3d.org/) for tasks related to ray sampling and ray marching.
