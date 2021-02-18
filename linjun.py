import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d


my_traj = np.load('my_traj.npy')
original_traj = np.load('/root/my_workspace/data/main_train/train/paths/2.npy')
mapp = np.load('/root/my_workspace/mpnet_dubins/saving_traj/map_v2.npy')
# print(my_traj)
print(mapp.shape)
