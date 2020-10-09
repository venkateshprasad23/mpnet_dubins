import os

import os.path as osp
import numpy as np

import math

count = 0
map_bounds = 16.3
volume = 10

saving_path_folder = '/root/my_workspace/data/modified_paths_retry/'
saving_costmap_folder = '/root/my_workspace/data/modified_costmaps_retry/'

mapping = np.load('map.npy')

print("hello")

def get_points(point):
    return point[0].x, point[0].y, point[0].z

# check_dist = lambda x,y: all(math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)<=2)

def check_dist(x,y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)<=1

def PossibleComb(points):
    '''
    A Function that returns a dictionry of all possible combinations of
    sub-trajectories, that can be broken into sub-sections, where the key
    of the dictionary is the starting index and the value is a list of all the
    other sub-indexes. A function check_dist that validates whether 2 points
    are withing the particular window is needed for this function.
    :param A list of points.
    :returns dict: A dictonary of all possible combinations.
    '''
    possible_comb = dict()
    global count
    for i, p1 in enumerate(points[:-1]):
        comb = [i]
        for j,pf in enumerate(points[i+1:]):
            if check_dist(get_points(p1),get_points(pf)):
                comb.append(j+i+1)
            else:
                break
        # for j, pb in enumerate(reversed(points[:i])):
        #     if check_dist(get_points(p1),get_points(pb)):
        #         comb.append(i-j-1)
        #     else:
        #         break
        possible_comb[i] = sorted(comb)

    path_array = []
    costmap_array = []

    # print(get_points(points[0]))

    print(possible_comb)
    
    for k in range(len(possible_comb)):
        if(len(possible_comb[k])>=3):
            # print(possible_comb[k])   
            path_array = []
            costmap_array = []

            for h in range(len(possible_comb[k])):
                path_array.append(get_points(points[possible_comb[k][h]]))
                print(possible_comb[k][h])
                costmap_array.append(get_costmap(points[possible_comb[k][h]]))
            
            # print(path_array)
            # print("\n")
            # print(costmap_array)
            count = count+1;
            path_array = np.array(path_array)
            costmap_array = np.array(costmap_array)

            # np.save(saving_path_folder + str(count) + '.npy',path_array)
            # np.save(saving_costmap_folder + str(count) + '.npy',costmap_array)
        else:
            continue

    # return possible_comb
def get_costmap(points):
    x,y,z = get_points(points)
    
    index_x = (x + map_bounds)/(0.2)
    index_x = index_x + 10
    index_x = int(index_x)

    index_y = (y + map_bounds)/(0.2)
    index_y = index_x + 10
    index_y = int(index_x)

    index_z = (z + map_bounds)/(0.2)
    index_z = index_x + 10
    index_z = int(index_x)

    costmap = mapping[index_x-volume:index_x+volume,index_y-volume:index_y+volume,index_z-volume:index_z+volume]

    return costmap

if __name__ == "__main__":
    trajFolder = '/root/paths/'
    print("hello")
    evlo = 0
    for entry in os.listdir(trajFolder):
        if '.npy' in entry:
            # s = int(entry.split(".")[0])
            # seeds.append(s)
            traj = np.load(osp.join(trajFolder,entry),allow_pickle=True)
            traj = np.reshape(traj,(traj.shape[0],1))
            # View trajectories from the perspective of the local costmap
            localtraj = np.copy(traj)
            evlo = evlo+1
            print(evlo)
            # print(PossibleComb(localtraj))
            PossibleComb(localtraj)




    