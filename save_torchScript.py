# Save the script of the model
import torch
import numpy as np

import voxel_ae as voxelNet
import model as model
from mpnet import MPnetBase
from misc import normalize, unnormalize


if __name__=="__main__":
    modelPath = '/root/my_workspace/data/new_trained_models/mpnet_epoch_49.pkl'
    saveTorchScriptModel = '/root/my_workspace/data/new_trained_models/mpnet_model_49.pt'

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
    
    sm = torch.jit.script(mpnet_base.mpNet)
    sm.save(saveTorchScriptModel)
