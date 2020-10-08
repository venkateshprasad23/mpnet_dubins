import os

import os.path as osp
import numpy as np

import math

count = 0
# map_bounds = 16.3
# volume = 10
def check_dist(x,y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)<=1

def see_dist(x,y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)

def get_points(point):
    # j=0
    # z = str(point[0])
    # lines = z.splitlines()
    # for j in range(3):
    #     h = lines[j]       

    #     if(j==0):
    #         u_x = float(h[4:len(h)])

    #     elif(j==1):
    #         u_y = float(h[4:len(h)])

    #     elif(j==2):
    #         u_z = float(h[4:len(h)])

    # return u_x, u_y, u_z
    return point[0].x, point[0].y, point[0].z

# trajFolder = '/root/my_workspace/data/modified_paths_retry/'

# path = np.load(traj_folder + '10.npy')

# for entry in os.listdir(trajFolder):
#     if '.npy' in entry:
#     	print("\n")
#         # s = int(entry.split(".")[0])
#         # seeds.append(s)
#         traj = np.load(osp.join(trajFolder,entry))
#         traj = np.reshape(traj,(traj.shape[0],1))
#         # View trajectories from the perspective of the local costmap
#         path = np.copy(traj)
#         print(path)
#         # evlo = evlo+1
#         # if(evlo==10):
#         #     break
#         # print(PossibleComb(localtraj))
#         # PossibleComb(localtraj)
#         for i in range(len(path)-1):
# 			print(check_dist(get_points(path[0]),get_points(path[i+1])))
# 			print(see_dist(get_points(path[0]),get_points(path[i+1])))
# 			print(path[0],path[i])

# for i in range(len(path)-1):
# 	print(check_dist(get_points(path[0]),get_points(path[i+1])))
# 	print(path[0],path[i+1])

# print(path.shape)

# mapping = np.load('map.npy')

# print("hello")

# def get_points(point):
#     j=0
#     z = str(point[0])
#     lines = z.splitlines()
#     for j in range(3):
#         h = lines[j]       

#         if(j==0):
#             u_x = float(h[4:len(h)])

#         elif(j==1):
#             u_y = float(h[4:len(h)])

#         elif(j==2):
#             u_z = float(h[4:len(h)])

#     return u_x, u_y, u_z


if __name__ == "__main__":
    trajFolder = '/root/my_workspace/data/modified_paths_retry/'
    # print("hello")
    count = 0
    saving_path_folder = '/root/my_workspace/data/ref_paths/'
    
    for entry in os.listdir(trajFolder):
        if '.npy' in entry:
            s = int(entry.split(".")[0])
            # seeds.append(s)
            path_array = []
            traj = np.load(osp.join(trajFolder,entry),allow_pickle=True)
            traj = np.reshape(traj,(traj.shape[0],1))
            # View trajectories from the perspective of the local costmap
            localtraj = np.copy(traj)
            # print(PossibleComb(localtraj))
            # PossibleComb(localtraj)
            start = localtraj[0]
            start_x, start_y, start_z = get_points(start)
            # path_array.append(start_x, start_y, start_z)
            count = count+1
            for points in localtraj:
                x,y,z = get_points(points)
                x = x - start_x
                y = y - start_y
                z = z - start_z
                path_array.append((x,y,z))
                print(x,y,z)

            np.save(saving_path_folder + str(s) + '.npy',path_array)
            print(count)
            # if(count==10):
            #     break



    