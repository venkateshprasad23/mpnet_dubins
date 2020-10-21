import numpy as np
import argparse
import torch

# import src.Model.AE.CAE as CAE_2d
import voxel_ae as voxelNet
import model as model


from misc import normalize, unnormalize
from train import MPnetTrain


def train(args):
    denormalize = unnormalize
    MLP = model.MLP
    network_parameters = {
        'normalize': normalize,
        'denormalize': denormalize,
        'encoderInputDim': [1, 40, 40, 40],
        'encoderOutputDim': 128,
        # 'worldSize': [27, 27, np.pi],
        'worldSize' : [1.73, 1.73, 1.73],
        'AE': voxelNet,
        'MLP': MLP,
        'modelPath': args.file,
    }
    
    trainNetwork = MPnetTrain(
        load_dataset=None,
        n_epochs=300,
        batchSize=32,
        opt=torch.optim.Adam,
        learning_rate=3e-4,
        **network_parameters
    )
    # trainNetwork.set_model_train_epoch(999)

    trainNetwork.train(numEnvsTrain=90000,
                       numEnvsTest=10000,
                       numPaths=1,
                       trainDataPath='/root/my_workspace/data/main_train/train/',
                       testDataPath='/root/my_workspace/data/main_train/test/')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
