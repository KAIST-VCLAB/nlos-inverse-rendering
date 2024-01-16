FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN conda update -n base -c defaults conda
RUN pip install opencv-python-headless opencv-contrib-python-headless tensorboard
RUN pip install -U matplotlib
RUN conda install -c anaconda h5py
RUN conda install -c conda-forge easydict
RUN pip install open3d -U
RUN apt-get install -y libqt5x11extras5
RUN pip install mat73