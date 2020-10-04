import os

import os.path as osp
import numpy as np

import math

count = 0
# map_bounds = 16.3
# volume = 10

saving_path_folder = '/root/my_workspace/data/ref_paths/'

# mapping = np.load('map.npy')

# print("hello")

def get_points(point):
    j=0
    z = str(point[0])
    lines = z.splitlines()
    for j in range(3):
        h = lines[j]       

        if(j==0):
            u_x = float(h[4:len(h)])

        elif(j==1):
            u_y = float(h[4:len(h)])

        elif(j==2):
            u_z = float(h[4:len(h)])

    return u_x, u_y, u_z


if __name__ == "__main__":
    trajFolder = '/root/my_workspace/data/modified_paths/'
    # print("hello")
    count = 0
    
    for entry in os.listdir(trajFolder):
        if '.npy' in entry:
            s = int(entry.split(".")[0])
            # seeds.append(s)
            path_array = []
            traj = np.load(osp.join(trajFolder,entry))
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

            np.save(saving_path_folder + str(s) + '.npy',path_array)
            print(count)
            # if(count==10):
            #     break



    