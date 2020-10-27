"""A class to load the point cloud data and target points in pixels """
import os
import re

import os.path as osp
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.utils.data

class ThreedIterDataset(torch.utils.data.IterableDataset):
    def __init__(self, folder_loc, numSamples):
        self.folder_loc = folder_loc
        # seeds = []
        our_dict = dict()
        count = 0
        numtraj = 1
        self.numSamples = numSamples

        for entry in os.listdir(osp.join(self.folder_loc,'paths')):
            if '.npy' in entry:
                s = int(entry.split(".")[0])
                shape = np.load(osp.join(self.folder_loc,'paths','{}.npy'.format(s))).shape[0] - 2
                our_dict[s] = list(range(count,count+shape))
                count = count + shape
                numtraj = numtraj+1
                if numtraj>=numSamples:
                    break
        self.seeds = list(range(1,count+1))
        self.our_dict = our_dict

    def get_key(self, val): 
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

        key = self.get_key(idx)
        values = self.our_dict[key]
        i = idx - values[0]

        costmap = np.load(osp.join(self.folder_loc,'costmaps','{}.npy'.format(key)),allow_pickle=True)
        traj = np.load(osp.join(self.folder_loc,'paths','{}.npy'.format(key)),allow_pickle=True)
        
        localtraj = np.copy(traj)

        obs = np.ones((1, 40, 40, 40))
        inputs = np.zeros((1, 6))
        targets = np.zeros((1, 3))

        goal = localtraj[-1]
        res = 0.2
        
        point = localtraj[i]
        mx, my, mz = round(point[0]/res), round(point[1]/res), round(point[2]/res),
        
        new_costmap = np.ones((40,40,40))
        new_costmap[10-mx:30-mx,10-my:30-my,10-mz:30-mz] = costmap
        
        obs[0,:,:,:] = new_costmap
        inputs = np.concatenate((localtraj[i], goal))
        targets = localtraj[i+1]    

        # return {
        #     'obs': np.array(obs, dtype=np.float32),
        #     'inputs': np.array(inputs, dtype=np.float32),
        #     'targets': np.array(targets, dtype=np.float32)
        # }

        return obs, inputs, targets
