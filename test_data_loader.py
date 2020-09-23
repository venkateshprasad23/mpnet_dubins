from torch.utils.data import DataLoader
from data_loader import ThreedDataset


if __name__ == "__main__":
    trainDataFileName = '/root/my_workspace/data/'
    train_ds = ThreedDataset(trainDataFileName, 10)
    train_dl = DataLoader(train_ds, shuffle=True, num_workers=5, batch_size=1, drop_last=True)

    count = 0
    # print(train_dl[5])
    for x in train_dl:
    	count+=1
    	print(count)
    	# print("\n")
    	# print(x)
