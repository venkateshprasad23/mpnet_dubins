from torch.utils.data import DataLoader
from my_data_loader import ThreedIterDataset


if __name__ == "__main__":
    trainDataFileName = '/root/my_workspace/data/main_train/train/'
    train_ds = ThreedIterDataset(trainDataFileName, 1000)
    train_dl = DataLoader(train_ds, num_workers=5, batch_size=256, drop_last=True)
    print("hello1")
    # count = 0

    for i_batch, sample_batched in enumerate(train_dl):
        print(i_batch, sample_batched['inputs'], sample_batched['targets'])

    # # print(train_ds)
    # # print(train_dl[5])
    # for batch in train_dl:
    #     print("hello2")
    #     # print(x)
    #     bobs, bi, bt = batch
    #     print(bobs,bi,bt)
    # 	# count+=1
    # 	# print(count)

    # 	# print("\n")
    # 	# print(x)
