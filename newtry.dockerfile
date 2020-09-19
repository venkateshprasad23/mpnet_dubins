# A docker container with devel files for ompl, torch and melodic
FROM jjj025/bionic_ompl:cuda10.1_devel

# Install ROS-melodic

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -q -y tzdata && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y\
  apt-utils \
  build-essential \
  psmisc \
  vim-gtk


# install packages
RUN apt-get update && apt-get install -q -y \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO melodic
# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# install ros-base packages
RUN apt-get update && apt-get install -y \
    ros-melodic-desktop-full=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

# Glx support files

RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libxau6 libxau6:i386 \
        libxdmcp6 libxdmcp6:i386 \
        libxcb1 libxcb1:i386 \
        libxext6 libxext6:i386 \
        libx11-6 libx11-6:i386 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    	libglvnd0 libglvnd0:i386 \
	libgl1 libgl1:i386 \
	libglx0 libglx0:i386 \
	libegl1 libegl1:i386 \
	libgles2 libgles2:i386 && \
    rm -rf /var/lib/apt/lists/*


# Setup for CUDA
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
  echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.0 brand=tesla,driver>=384,driver<385"

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Install dependcies for rosbag for python3
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install rospkg

# COPY ./requirements.txt /root/

WORKDIR /root/
# RUN pip3 install -r requirements.txt
RUN pip3 install matplotlib pandas jupyter torch torchvision gnupg

# Install rc_car files
RUN apt-get update && apt-get install -y \
    ros-melodic-velodyne \
    ros-melodic-ackermann-msgs \
    ros-melodic-joy \
    ros-melodic-serial

# WORKDIR /root/racecar/src
# RUN git clone https://github.com/mit-rss/wall_follower_sim.git
# RUN git clone https://github.com/mit-racecar/racecar_simulator.git

# RUN git clone https://github.com/mit-racecar/racecar.git

RUN apt-get install -y \
    ros-melodic-tf2-geometry-msgs \
    ros-melodic-map-server

# WORKDIR /root/racecar

# Install hybrid dubin-curves
RUN apt-get install -y \
    libeigen3-dev \
    ros-melodic-navigation

RUN apt-get install -y \
    wget \
    gfortran \
    cppad
    

RUN echo "source /opt/ros/melodic/setup.bash" >> /etc/.bashrc
# WORKDIR /root/racecar
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash"

# RUN git clone https://github.com/venkateshprasad23/mpnet_local_planner.git


CMD ["bash"]
