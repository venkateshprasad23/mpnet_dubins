import csv
import matplotlib.pyplot as plt
import numpy as np

datafile = open('/root/my_workspace/data/trained_models/progress.csv', 'r')
myreader = csv.reader(datafile)

array_test = []
array_train = []
# array = np.array(array)

for row in myreader:
	# print(row[1])
	# print(row[0])
	if(row[1] == 'train_loss'):
		continue
	array_test.append(float(row[0]))
	array_train.append(float(row[1]))
# print(array)
plt.plot(array_train,'r')
plt.plot(array_test,'g')
plt.savefig('losses.png')
    