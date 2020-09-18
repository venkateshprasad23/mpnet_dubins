import os

import os.path as osp
import numpy as np

import math

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

# check_dist = lambda x,y: all(math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)<=2)

def check_dist(x,y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)<=2

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
    for i, p1 in enumerate(points[:-1]):
        comb = [i]
        for j,pf in enumerate(points[i+1:]):
            if check_dist(get_points(p1),get_points(pf)):
                comb.append(j+i+1)
            else:
                break
        for j, pb in enumerate(reversed(points[:i])):
            if check_dist(get_points(p1),get_points(pb)):
                comb.append(i-j-1)
            else:
                break
        possible_comb[i] = sorted(comb)
    return possible_comb

if __name__ == "__main__":
    trajFolder = '/root/my_workspace/data/paths/'
    for entry in os.listdir(trajFolder):
        if '.npy' in entry:
            # s = int(entry.split(".")[0])
            # seeds.append(s)
            traj = np.load(osp.join(trajFolder,entry))
            traj = np.reshape(traj,(traj.shape[0],1))
            # View trajectories from the perspective of the local costmap
            localtraj = np.copy(traj)
            print(PossibleComb(localtraj))

    


    