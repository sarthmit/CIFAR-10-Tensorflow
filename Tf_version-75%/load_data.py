import numpy as np
import cPickle
import tensorflow as tf

def unpickle(file):
	with open(file , 'rb') as fo:
		dict = cPickle.load(fo)
	dict["data"] = dict["data"].reshape(-1,3,32,32)
	dict["data"] = dict["data"].transpose(0,2,3,1)
	dict["data"] = dict["data"].reshape(-1,3072)
	return dict

def convert_one_hot(data,size):
	data_size = np.shape(data)
	final = np.zeros((data_size[0],size))
	final[np.arange(data_size[0]), data] = 1
	return final

def next_batch(dict,size,shuf,start=0):
	arr = np.arange(50000)
	data_size = np.shape(dict["data"])
	label_size = np.shape(dict["labels"])
	final = (data_size[0] , (data_size[1] + label_size[1]))
	data = np.zeros(final)
	data[:,:data_size[1]] = dict["data"]
	data[:,data_size[1]:] = dict["labels"]
	if shuf==1:
		np.random.shuffle(arr)
	return data[arr[start:start+size],:data_size[1]], data[arr[start:start+size],data_size[1]:]

def GCN(data):
	data = np.array(data)
	data = (data - np.mean(data,axis=1)[:,np.newaxis])/ np.std(data,axis=1)[:,np.newaxis]
	return data

def zca_whitening(data):
	sigma = np.cov(np.transpose(data))
	U,S,V = np.linalg.svd(sigma)
	epsilon = 1e-5
	ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
	return np.dot(data,ZCAMatrix)

def create_data():
	batch1=unpickle('cifar-10-batches-py/data_batch_1')
	batch2=unpickle('cifar-10-batches-py/data_batch_2')
	batch3=unpickle('cifar-10-batches-py/data_batch_3')
	batch4=unpickle('cifar-10-batches-py/data_batch_4')
	batch5=unpickle('cifar-10-batches-py/data_batch_5')

	batch1_data=batch1['data']
	batch2_data=batch2['data']
	batch3_data=batch3['data']
	batch4_data=batch4['data']
	batch5_data=batch5['data']

	batch1_labels=batch1['labels']
	batch2_labels=batch2['labels']
	batch3_labels=batch3['labels']
	batch4_labels=batch4['labels']
	batch5_labels=batch5['labels']

	train_data = np.concatenate((batch1_data,batch2_data,batch3_data,batch4_data,batch5_data), axis=0)
	train_labels = np.concatenate((batch1_labels,batch2_labels,batch3_labels,batch4_labels,batch5_labels),axis=0)

	return {"data":train_data, "labels":train_labels}