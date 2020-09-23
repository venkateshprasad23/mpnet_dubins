import shutil
import os

source_paths = '/root/my_workspace/data/modified_paths/'
source_costmaps = '/root/my_workspace/data/modified_costmaps/'

destination_train_paths = '~/my_workspace/data/main_train/train/paths/'
destination_train_costmaps = '~/my_workspace/data/main_train/train/costmaps/'

destination_test_paths = '/root/my_workspace/data/main_train/test/paths/'
destination_test_costmaps = '/root/my_workspace/data/main_train/test/costmaps/'

files_paths = os.listdir(source_paths)
files_costmaps = os.listdir(source_costmaps)

print(files_paths)
print(files_costmaps)
# train_paths
count = 1
for file in files_paths:
	print("inside for loop")
	if(count<=20000):
		print("inside first if condition")
		shutil.copy(file, destination_train_paths)
		count+=1
		print(count)
		print("\n")
	elif(count>20000 and count<=25000):
		print("inside second if condition")
		shutil.copy(file, destination_test_paths)
		count+=1
	else:
		break

count = 1
for file in files_costmaps:
	if(count>=1 and count<=20000):
		shutil.copy(file, destination_train_costmaps)
		count+=1
	elif(count>20000 and count<=25000):
		shutil.copy(file, destination_test_costmaps)
		count+=1
	else:
		break

	# print(new_path)