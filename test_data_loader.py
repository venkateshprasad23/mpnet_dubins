from torch.utils.data import DataLoader
from my_data_loader import ThreedDataset, ThreedIterDataset


if __name__ == "__main__":
    trainDataFileName = '/root/my_workspace/data/main_train/train/'
    train_ds = ThreedIterDataset(trainDataFileName)
    train_dl = DataLoader(train_ds, shuffle=True, num_workers=5, batch_size=16, drop_last=True)

    count = 0
    # print(train_dl[5])
    for batch in train_dl:
        bobs, bi, bt = batch
        print(bobs,bi,bt)
    	# count+=1
    	# print(count)

    	# print("\n")
    	# print(x)
