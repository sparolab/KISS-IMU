# Temporary Dockerfile for KISS-IMU release.
# The base image already ships with ROS Noetic, CUDA, PyTorch, pypose,
# kiss-icp, small_gicp, pygicp, scikit-learn, etc., so we don't reinstall.
#
# When you (the user) host this image under your lab's account, replace
# the FROM line below.
FROM jivvon1/ros-noetic-cuda:v3

# QoL: avoid X11 MIT-SHM for matplotlib/cv2 windows
ENV QT_X11_NO_MITSHM=1

# Default workdir matches the volume mount in docker-compose.yml
WORKDIR /home/test_ws/src

# (No pip install — base image already has everything needed.
# If you ever rebuild from a leaner base, install these:
#   pip install pypose scikit-learn open3d small_gicp pygicp kiss-icp tensorboard tqdm)

CMD ["/bin/bash"]
