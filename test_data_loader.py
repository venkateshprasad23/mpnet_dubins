from torch.utils.data import DataLoader
from my_data_loader import ThreedDataset, ThreedIterDataset


if __name__ == "__main__":
    trainDataFileName = '/root/my_workspace/data/main_train/train/'
    train_ds = ThreedIterDataset(trainDataFileName)
    train_dl = DataLoader(train_ds, num_workers=5, batch_size=16, drop_last=True)
    print("hello1")
    count = 0

    print(train_dl)
    # print(train_dl[5])
    for x in train_dl:
        print("hello2")
        print(x)
        bobs, bi, bt = batch
        print(bobs,bi,bt)
    	# count+=1
    	# print(count)

    	# print("\n")
    	# print(x)
