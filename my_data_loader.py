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


def CenterRobot(costmap, pixel_ind):
    costmap_data = costmap.get_data()
    costmap_dim = costmap_data.shape
    full_obs = np.ones((costmap_dim[0] * 2, costmap_dim[1] * 2))
    x_0, y_0 = costmap_dim[1] - pixel_ind[1], costmap_dim[0] - pixel_ind[0]
    full_obs[x_0:x_0 + costmap_dim[1], y_0:y_0 +
             costmap_dim[0]] = costmap_data / 254
    full_obs = full_obs[::3, ::3]
    full_obs = torch.Tensor(full_obs).unsqueeze(0)
    return full_obs

class ThreedDataset(torch.utils.data.Dataset):
    def __init__(self, folder_loc, numSamples):
        self.folder_loc = folder_loc
        self.numSamples = numSamples
        self.inputs = np.zeros((numSamples, 6))
        self.targets = np.zeros((numSamples, 3))
        self.obs = np.zeros((numSamples, 1, 40, 40, 40))
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
    def __init__(self, folder_loc):
        self.folder_loc = folder_loc
        seeds = []
        our_dict = dict()
        count = 0

        for entry in os.listdir(folder_loc):
            if '.npy' in entry:
                s = int(entry.split(".")[0])
                shape = np.load(osp.join(self.folder_loc,'paths','{}.npy'.format(s))).shape[0] - 1
                our_dict[s] = [x for x in range(count+1,count+shape)]
                count = count + shape
                seeds = seeds + [x for x in range(count+1,count+shape)]
        self.seeds = seeds
        self.our_dict = our_dict

    def get_key(val): 
        for key, value in self.our_dict.items(): 
            if val in value:
                return key  

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

        key = get_key(idx)
        values = self.our_dict[key]
        i = idx - values[0]

        # resl = msg.info.resolution
        # x0, y0 = msg.info.origin.position.x, msg.info.origin.position.y
        costmap = np.load(osp.join(self.folder_loc,'costmaps','{}.npy'.format(key)),allow_pickle=True)
        traj = np.load(osp.join(self.folder_loc,'paths','{}.npy'.format(key)),allow_pickle=True)
        # traj = np.reshape(traj,(traj.shape[0],1))
        # View trajectories from the perspective of the local costmap
        localtraj = np.copy(traj)
        # localtraj[:,:2] = localtraj[:,:2] - np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        # samples = traj.shape[0] - 1

        obs = np.ones((1, 40, 40, 40))
        inputs = np.zeros((1, 6))
        targets = np.zeros((1, 3))
        # j = 0

        # costmap = np.array(msg.data).reshape(msg.info.height,msg.info.width)
        # for t in traj:
        #     t[2] = normalize_angle(t[2])
        goal = localtraj[-1]
        res = 0.2
        # goal = world_to_voxel(goal)

        # for i in range(samples):
        # for i, point in enumerate(traj[:-1]):
        point = localtraj[i]
        mx, my, mz = round(point[0]/res), round(point[1]/res), round(point[2]/res),
        # if 0>mx or mx>120 or 0>my or my>120:
        #     print(mx, my, idx)
        #     return {'obs': [], 'inputs': [], 'targets': []}
        new_costmap = np.ones((40,40,40))
        new_costmap[10-mx:30-mx,10-my:30-my,10-mz:30-mz] = costmap
        # Normalize and compress the image by 3 times
        obs[0,:,:,:] = new_costmap
        inputs = np.concatenate((localtraj[i], goal))
        targets = localtraj[i+1]    

        # return {
        #     'obs': np.array(obs, dtype=np.float32),
        #     'inputs': np.array(inputs, dtype=np.float32),
        #     'targets': np.array(targets, dtype=np.float32)
        # }

        return obs, inputs, targets
