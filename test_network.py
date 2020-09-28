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

def format_data(self, obs, inputs, targets):
        """
        Formats the data to be fed into the neural network
        """
        bobs, bi = self.format_input(obs, inputs)
        # Format targets
        if isinstance(targets, np.ndarray):
            bt = torch.FloatTensor(targets)
        else:
            bt = targets.float()
        bt = self.normalize(bt, self.worldSize)
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
    testObs, testInput, testTarget = test_ds[:int(5)]
    testObs, testInput, testTarget = format_data(
        testObs, testInput, testTarget)

    with torch.no_grad():
        # test loss
        network_output = self.mpNet(testInput, testObs)
        # test_loss_i = self.mpNet.loss(
        #     network_output,
        #     testTarget
        #     ).sum(dim=1).mean()
        # test_loss_i = get_numpy(test_loss_i)
        print("Network Output:")
        print(network_output)
        print("\n")
        print("Test Target:")
        print(testTarget)
        print("\n")

