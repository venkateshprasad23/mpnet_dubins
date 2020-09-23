import shutil
import os

source_paths = "/root/my_workspace/data/modified_paths/"
source_costmaps = "/root/my_workspace/data/modified_costmaps/"

destination_train_paths = "/root/my_workspace/data/main_train/train/paths/"
destination_train_costmaps = "/root/my_workspace/data/main_train/train/paths/"

destination_test_paths = "/root/my_workspace/data/main_train/test/paths/"
destination_test_costmaps = "/root/my_workspace/data/main_train/test/paths/"

files_paths = os.listdir(source_paths)
files_costmaps = os.listdir(source_costmaps)

# train_paths
count = 1
for file in files_paths:
	if(count>=1 && count<=20000):
		shutil.copy(f"{source_paths}/{file}", destination_train_paths)
		count+=1
	elif(count>20000 && count<=25000):
		shutil.copy(f"{source_paths}/{file}", destination_test_paths)
		count+=1
	else:
		break

count = 1
for file in files_costmaps:
	if(count>=1 && count<=20000):
		shutil.copy(f"{source_costmaps}/{file}", destination_train_costmaps)
		count+=1
	elif(count>20000 && count<=25000):
		shutil.copy(f"{source_costmaps}/{file}", destination_test_costmaps)
		count+=1
	else:
		break

	# print(new_path)