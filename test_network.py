# Save the script of the model
import torch
import numpy as np

import voxel_ae as voxelNet
import model as model
from mpnet import MPnetBase
from misc import normalize, unnormalize

from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

from data_loader import ThreedDataset

from misc import load_net_state, load_opt_state, save_state, to_var, load_seed

worldSize = [2, 2, 2]

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
    modelPath = '/root/my_workspace/data/trained_models/mpnet_epoch_299.pkl'
    testDataPath='/root/my_workspace/data/test_network/'
    # saveTorchScriptModel = '/root/data/grid_world_2_0_06/trained_models/mpnet_model_289.pt'

    network_param = {
        "normalize": normalize,
        "denormalize": unnormalize,
        "encoderInputDim": [1, 20, 20, 20],
        "encoderOutputDim": 128,
        "worldSize": [2, 2, 2],
        "AE": voxelNet,
        "MLP": model.MLP,
        "modelPath": modelPath}

    mpnet_base = MPnetBase(**network_param)
    mpnet_base.load_network_parameters(modelPath)
    
    # sm = torch.jit.script(mpnet_base.mpNet)
    # sm.save(saveTorchScriptModel)
    test_ds = ThreedDataset(testDataPath, 10)
    testObs, testInput, testTarget = test_ds[:10]
    testObs, testInput, testTarget = format_data(
        testObs, testInput, testTarget)

    if torch.cuda.is_available():
            mpnet_base.mpNet.cuda()
            mpnet_base.mpNet.mlp.cuda()
            mpnet_base.mpNet.encoder.cuda()

    with torch.no_grad():
        # test loss
        network_output = mpnet_base.mpNet(testInput, testObs)
        # network_output = unnormalize(network_output,worldSize)
        test_loss_i = mpnet_base.mpNet.loss(
            network_output,
            testTarget
            ).sum(dim=1).mean()
        test_loss_i = get_numpy(test_loss_i)
        print("Network Output:")
        print(network_output)
        print("\n")
        print("Test Target:")
        print(testTarget)
        print("\n")

