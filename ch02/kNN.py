#!/usr/bin/env python3
from numpy import *
import operator

def create_dataset():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels


def classify0(inx, dataset, labels, k):
	dataset_size = dataset.shape[0]
	diff_mat = tile(inx, (dataset_size,1)) - dataset
	sq_diff_mat = diff_mat ** 2
	sq_distances = sq_diff_mat.sum(axis=1)
	distances = sq_distances ** 0.5

	sorted_dist = distances.argsort()

	class_count = {}

	for i in range(k):
		vote_label = labels[sorted_dist[i]]
		class_count[vote_label] = class_count.get(vote_label, 0) + 1

	sorted_class_count = sorted(class_count.items(),\
		key= operator.itemgetter(1), reverse=True)

	return sorted_class_count[0][0]


def file2matrix(filename):
	fr = open(filename)
	array_lines = fr.readlines()
	n_lines = len(array_lines)
	return_mat = zeros((n_lines, 3))

	class_label_vector = []
	index = 0

	for line in array_lines:
		line = line.strip()
		list_from_line = line.split('\t')
		return_mat[index,:] = list_from_line[0:3]
		class_label_vector.append(int(list_from_line[-1]))
		index += 1

	return return_mat, class_label_vector




if __name__ == '__main__':

	'''
	group, labels = create_dataset()
	inx = [0, 0]
	result = classify0(inx, group, labels, 3)
	print("result: ", result
	'''

	dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
	print(dating_data_mat)
	print(dating_labels[0:20])

	import matplotlib
	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(dating_data_mat[:,0], dating_data_mat[:,1],\
		15.0*array(dating_labels), 15.0*array(dating_labels))
	plt.show()

