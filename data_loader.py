"""A class to load the point cloud data and target points in pixels """
import os
import re

import os.path as osp
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.utils.data

# import rosbag

# def normalize_angle(z):
#     """
#     A function to wrap around -1 and 1
#     """
#     return (z + np.pi) % (2 * np.pi) - np.pi


def get_points(point):
    '''
    Converts world co-ordinates to pixel co-ordinates
    :param point: The desired point to be converted
    :param x0: The x co-ordinate of the image origin
    :param y0: The y co-ordinate of the image origin
    :param resl: The resolution of the costmap
    :returns [mx,my] index corresponding to the image 
    '''
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

class ThreedDataset(torch.utils.data.Dataset):
    def __init__(self, folder_loc, numSamples):
        self.folder_loc = folder_loc
        self.numSamples = numSamples
        self.inputs = np.zeros((numSamples, 6))
        self.targets = np.zeros((numSamples, 3))
        self.obs = np.zeros((numSamples, 1, 20, 20, 20))
        i = 0
        done = False

        trajFolder = osp.join(folder_loc, 'paths')
        seeds = []

        for entry in os.listdir(trajFolder):
            if '.npy' in entry:
                s = int(entry.split(".")[0])
                seeds.append(s)

        DataSet = ThreedIterDataset(folder_loc, seeds)
        Data = DataLoader(DataSet, num_workers=5)

        # if not seeds:
        #     raise ValueError("{} - Not a valid folder".format(trajFolder))
        # Load point cloud, points and target information
        count = 0
        for data in Data:
            if len(data['obs'])==0:
                count +=1
                continue

            numSubSamples = data['obs'].shape[1]

            stop_iter = min(i+numSubSamples, numSamples)
            self.obs[i:stop_iter, ...] = data['obs'].squeeze(0)[:stop_iter-i,...]
            self.inputs[i:stop_iter, ...] = data['inputs'].squeeze(0)[:stop_iter-i,...]
            self.targets[i:stop_iter, ...] = data['targets'].squeeze(0)[:stop_iter-i,...]
            i = stop_iter
            if i == numSamples:
                done = True
                break

    def __len__(self):
        return self.numSamples

    def __getitem__(self, idx):
        return self.obs[idx, ...], self.inputs[idx, ...], self.targets[idx, ...]


class ThreedIterDataset(torch.utils.data.IterableDataset):
    def __init__(self, folder_loc, seeds):
        self.folder_loc = folder_loc
        self.seeds = seeds

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.seeds)
        else:
            per_worker = int(len(self.seeds) // worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.seeds))

        return iter(self.GetItem(s) for s in self.seeds[iter_start:iter_end])

    def GetItem(self, idx):
        # with rosbag.Bag(osp.join(self.folder_loc, 'costmap','costmap_{}.bag'.format(idx))) as rosbagObject:
        #     bagItem, = list(rosbagObject.read_messages('lcm'))
        #     _, msg, t = bagItem

        # resl = msg.info.resolution
        # x0, y0 = msg.info.origin.position.x, msg.info.origin.position.y
        costmap = np.load(osp.join(self.folder_loc,'costmaps','{}.npy'.format(idx)),allow_pickle=True)
        traj = np.load(osp.join(self.folder_loc,'paths','{}.npy'.format(idx)),allow_pickle=True)
        # traj = np.reshape(traj,(traj.shape[0],1))
        # View trajectories from the perspective of the local costmap
        localtraj = np.copy(traj)
        # localtraj[:,:2] = localtraj[:,:2] - np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        samples = traj.shape[0] - 1

        obs = np.ones((samples, 1, 20, 20, 20))
        inputs = np.zeros((samples, 6))
        targets = np.zeros((samples, 3))
        # j = 0

        # costmap = np.array(msg.data).reshape(msg.info.height,msg.info.width)
        # for t in traj:
        #     t[2] = normalize_angle(t[2])
        goal = localtraj[-1]

        # goal = world_to_voxel(goal)

        for i in range(samples):

            # mx, my = world_to_pixel(point, x0, y0, resl)
            # if 0>mx or mx>120 or 0>my or my>120:
            #     print(mx, my, idx)
            #     return {'obs': [], 'inputs': [], 'targets': []}
            # new_costmap = np.ones((120*2,120*2))*100
            # new_costmap[120-my:240-my,120-mx:240-mx] = costmap
            # Normalize and compress the image by 3 times
            obs[i,0,:,:,:] = costmap[i]
            inputs[i, :] = np.concatenate((localtraj[i], goal))
            targets[i, :] = localtraj[i+1]

        return {
            'obs': np.array(obs, dtype=np.float32),
            'inputs': np.array(inputs, dtype=np.float32),
            'targets': np.array(targets, dtype=np.float32)
        }
