# Save the script of the model
import torch
import numpy as np
import os.path as osp

import voxel_ae as voxelNet
import model as model
from mpnet import MPnetBase
from misc import normalize, unnormalize

from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

from data_loader import ThreedDataset

from misc import load_net_state, load_opt_state, save_state, to_var, load_seed

worldSize = [1.73, 1.73, 1.73]

get_numpy = lambda x: x.data.cpu().numpy()

def format_input(obs, inputs):
        """
        Formats the input data that needed to be fed into the network
        """
        if isinstance(inputs, np.ndarray):
            bi = torch.FloatTensor(inputs)
        else:
            bi = inputs.float()
        if isinstance(obs, np.ndarray):
            bobs = torch.FloatTensor(obs)
        else:
            bobs = obs.float()

        # Normalize observations
        # normObsVoxel = torchvision.transforms.Normalize([0.5], [1])
        # for i in range(bobs.shape[0]):
        #     bobs[i, ...] = normObsVoxel(bobs[i, ...])
        bi = normalize(bi, worldSize)
        return to_var(bobs), to_var(bi)

def format_data(obs, inputs, targets):
        """
        Formats the data to be fed into the neural network
        """
        bobs, bi = format_input(obs, inputs)
        # Format targets
        if isinstance(targets, np.ndarray):
            bt = torch.FloatTensor(targets)
        else:
            bt = targets.float()
        bt = normalize(bt, worldSize)
        bt = to_var(bt)
        return bobs, bi, bt


if __name__=="__main__":
    modelPath = '/root/my_workspace/data/new_trained_models/mpnet_epoch_99.pkl'
    testDataPath='/root/my_workspace/data/test_network/'
    folder_loc = '/root/my_workspace/data/main_train/train/'

    network_param = {
        "normalize": normalize,
        "denormalize": unnormalize,
        "encoderInputDim": [40, 40, 40],
        "encoderOutputDim": 128,
        "worldSize": [1.73, 1.73, 1.73],
        "AE": voxelNet,
        "MLP": model.MLP,
        "modelPath": modelPath}

    mpnet_base = MPnetBase(**network_param)
    mpnet_base.load_network_parameters(modelPath)

    if torch.cuda.is_available():
        print("CUDA is available!")
        mpnet_base.mpNet.cuda()
        mpnet_base.mpNet.mlp.cuda()
        mpnet_base.mpNet.encoder.cuda()

    idx = 2
    costmap = np.load(osp.join(folder_loc,'costmaps','{}.npy'.format(idx)))
    traj = np.load(osp.join(folder_loc,'paths','{}.npy'.format(idx)))

    print(traj)

    start = traj[0]
    
    print("Initial start, before reshaping: ",start)
    print("Shape: ",start.shape)
    # start = torch.tensor(start).float().reshape(1,-1)
    print("Start, after reshaping: ",start)
    print("Shape: ",start.shape)
    goal = traj[-1]
    print("Initial goal, before reshaping: ",goal)
    print("Shape: ",goal.shape)
    # goal = torch.tensor(goal).float().reshape(1,-1)
    print("Goal, after reshaping: ",goal)
    print("Shape: ",goal.shape)

    dist = np.linalg.norm(start-goal)
    print(dist)
    current = start
    res = 0.2

    while((np.linalg.norm(current-goal)) > 0.2):
        mx, my, mz = round(current[0]/res), round(current[1]/res), round(current[2]/res)
        new_costmap = np.ones((40,40,40))
        new_costmap[10-mx:30-mx,10-my:30-my,10-mz:30-mz] = costmap

        obs = np.ones((1, 40, 40, 40))
        obs[0, :,:,:] = new_costmap
        obs = torch.Tensor(obs)

        current = torch.tensor(current).float().reshape(1,-1)
        goal = torch.tensor(goal).float().reshape(1,-1)
        network_input = torch.cat((current,goal), dim=1)

        tobs, tInput = format_input(obs, network_input)
        temp = mpnet_base.mpNet(tInput, tobs).data.cpu() 
        temp = unnormalize(temp.squeeze(), worldSize)
        current = temp
        print(current)
    
        

    # mx, my, mz = 0, 0, 0#round(point[0]/res), round(point[1]/res), round(point[2]/res)
    # new_costmap = np.ones((40,40,40))
    # new_costmap[10-mx:30-mx,10-my:30-my,10-mz:30-mz] = costmap

    # obs = np.ones((1, 40, 40, 40))
    # obs[0, :,:,:] = new_costmap
    # obs = torch.Tensor(obs)

    # network_input = torch.cat((start,goal), dim=1)
    # # print("Network Input: ",network_input)
    # # print("Shape: ",network_input.shape)
    # tobs, tInput = format_input(obs, network_input)
    # # print("tInput: ",tInput)
    # # print("Unsqueezed shape: ", tobs.shape)
    # temp = mpnet_base.mpNet(tInput, tobs).data.cpu()
    # print("Before normalization : ", temp)
    # temp = unnormalize(temp.squeeze(), worldSize)
    # # temp = start + temp

    # print('Network Output : {}, trajectory value: {}'.format(temp, traj[:,:]))





    

    
