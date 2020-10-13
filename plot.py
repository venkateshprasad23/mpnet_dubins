import csv
import matplotlib.pyplot as plt
import numpy as np

datafile = open('/root/my_workspace/data/trained_models/progress.csv', 'r')
myreader = csv.reader(datafile)

array = []
# array = np.array(array)

for row in myreader:
	# print(row[1])
	# print(row[0])
	if(row[1] == 'train_loss'):
		continue
	array.append(float(row[0]))
# print(array)
plt.plot(array)
plt.savefig('test_loss.png')
    