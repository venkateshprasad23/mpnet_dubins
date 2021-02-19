import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from mpl_toolkits import mplot3d


my_traj = np.load('my_traj.npy')
original_traj = np.load('/root/my_workspace/data/main_train/test/paths/2.npy')
mapp = np.load('/root/my_workspace/mpnet_dubins/saving_traj/map_v2.npy')
# print(my_traj)
# print(mapp.shape)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(mapp)

# o3d.visualization.draw_geometries([pcd])

fig = plt.figure()
ax = plt.axes(projection='3d')

# x = my_traj[:,0]
# y = my_traj[:,1]
# z = my_traj[:,2]

# x1 = original_traj[:,0]
# y1 = original_traj[:,1]
# z1 = original_traj[:,2]
x1 = [a[0] for a in original_traj]
y1 = [a[1] for a in original_traj]
z1 = [a[2] for a in original_traj]

x = [a[0] for a in my_traj]
y = [a[1] for a in my_traj]
z = [a[2] for a in my_traj]



# ax.scatter3D(x1,y1,z1,cmap='Greens')
plt.plot(x1,y1,z1)
plt.plot(x,y,z)
ax.scatter3D(original_traj[0][0], original_traj[0][1], original_traj[0][2])

plt.savefig('my_traj.png')

