import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d############# Load trajectory
prefix = './data'
hash_set = np.load("{}/hash.npy".format(prefix), allow_pickle=True, encoding='bytes').tolist()
hash_set2 = hash_set.copy()
def load_traj(start_goal_hash, prefix='/media/arclabdl1/HD1/Linjun/catkin_ws/data'):
    # print("{}/traj_{}.npy".format(prefix, str(start_goal_hash)[1:-1].replace(", ", "_")))
    return np.loadtxt("{}/traj_{}.txt".format(prefix, str(start_goal_hash)[1:-1].replace(", ", "_")), delimiter=" ")############# Plot
xyz = np.load("pcl.npy")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)line_sets = []l = list(hash_set)
# print(l)
traj_id = 0
try:
    traj = load_traj(l[traj_id], prefix=prefix)
    if len(traj) > len(longest):
        # print(len(traj), len(longest))
        longest = traj
except:
    print(traj_id)
    pass
    # points = longest
points = traj[:, :3]
lines = [[i, i+1] for i in range(len(traj) - 1)]line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)
# line_set.colors = o3d.utility.Vector3dVector(colors)
line_sets.append(line_set)o3d.visualization.draw_geometries([pcd] + line_sets)