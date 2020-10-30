import numpy as np
import argparse
import torch

# import src.Model.AE.CAE as CAE_2d
import voxel_ae as voxelNet
import model as model


from misc import normalize, unnormalize
from train import MPnetTrain

torch.cuda.empty_cache()

def debug_memory():
    import collections, gc, resource
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))

# torch.set_default_tensor_type(torch.cuda.HalfTensor)
# torch.multiprocessing.set_start_method('spawn',force=True)

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
        batchSize=256,
        opt=torch.optim.Adam,
        learning_rate=3e-4,
        **network_parameters
    )
    trainNetwork.set_model_train_epoch(119)
    # debug_memory()
    trainNetwork.train(numEnvsTrain=150000,
                       numEnvsTest=1000,
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
