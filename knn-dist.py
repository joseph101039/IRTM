#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import numpy as np
import theano.tensor as T
import theano
import time
import sys

class Timer(object):
	def __init__(self):
		self.startTime = time.time()
		self.curTime = time.time()
	def getTime(self):
		self.curTime = time.time()
		return "Total time: %f sec" %(self.curTime - self.startTime)
	def getTimeGap(self):
		t = time.time()
		gap = t - self.curTime
		self.curTime = t
		return " %f sec" %(gap)	

tokenized_doc_loc = "/media/joseph/SSD/TrainingData_docVec_jieba/"

def knn_training(file_path):
	superClass = ["運動","娛樂","生活","社會","地方","要聞","兩岸","全球","產經","股市", "即時"]
	subClass = ["運動","娛樂","生活","社會","地方","要聞","兩岸","國際","財經"]

	with open(file_path, 'r') as dr:
		doc_list = dr.read().splitlines()
		doc_list = [x.split(' ') for x in doc_list]

	#caculate normalized term-frequency of features for all training documents
	train_vec = []
	for i in range(len(doc_list)):
		with open(tokenized_doc_loc + doc_list[i][0], 'r') as dr:
			vec = dr.read().split(' ')
			vec = [float(k) for k in vec]
			train_vec.append(vec)

	train_vec = np.asmatrix(train_vec, dtype = 'float32')

	train_class = []
	for i in range(len(doc_list)):
		sup_index = superClass.index(doc_list[i][1])
		if sup_index == 10: 
			sub_index = subClass.index(doc_list[i][2])
			train_class.append(sub_index)
		else:
			train_class.append(sup_index)
	return [train_vec, train_class]


def knn_testing(file_path, _knn_test, train_class, K_NUM):
	superClass = ["運動","娛樂","生活","社會","地方","要聞","兩岸","全球","產經","股市", "即時"]
	subClass = ["運動","娛樂","生活","社會","地方","要聞","兩岸","國際","財經"]
	
	with open(file_path, 'r') as dr:
		doc_list = dr.read().splitlines()
		doc_list = [x.split(' ') for x in doc_list]

	#produce target output (real classes for testing document)
	target_out = []
	for i in range(len(doc_list)):
		sup_index = superClass.index(doc_list[i][1])
		if sup_index == 10: 
			sub_index = subClass.index(doc_list[i][2])
			target_out.append(sub_index)
		else:
			target_out.append(sup_index)

	#produce predict output (predict classes for testing document)
	predict_out = []
	drange = int(len(doc_list) / 50)
	correct_num, total_num = [0., 0.]
	test_vec = []
	for i in range(len(doc_list)):
		with open(tokenized_doc_loc + doc_list[i][0], 'r') as dr:
			test_vec = dr.read().split(' ')
			test_vec = [float(k) for k in test_vec]

		test_vec = np.asarray(test_vec, dtype = 'float32')
		y = _knn_test(test_vec)
		y_index = y.argsort()[:K_NUM]

		vote_class = np.full(10, 0, 'int')
		for index in y_index:
			vote_class[train_class[index]] += 1

		predict_class = np.argmax(vote_class)
		predict_out.append(predict_class)

		if predict_class == target_out[i]:
			correct_num += 1
		total_num += 1
	
		if i % drange == 0:
			p = int(i / drange) 
			sys.stdout.write('\r')
			sys.stdout.write("[%-51s] %d%% accu: %f" % ('='*p+'>', 2*p, (correct_num / total_num)))
			sys.stdout.flush()
			correct_num, total_num = [0., 0.]

	return [predict_out, target_out]


if __name__ == '__main__':
	train_file = ['100t', '50t', '20t', '10t', '5t', '2t', '1t']
	for tname in train_file:
		K_NUM = 15

		timer = Timer()	###
		train_vec, train_class =  knn_training('../svm/doc_result/doc_merge_vec_train_' + tname + '.txt')
		print 'doc_merge_vec_train_' + tname + '.txt'

		#theano function declaration
		ntv = theano.shared(np.asarray(train_vec, dtype = 'float32'))
		train_vec = None
		x = T.vector(dtype="float32")
		y = T.power(ntv- x, 2).sum(axis=1)
		_knn_test = theano.function(inputs = [x], outputs = y)

		predict_out, target_out = knn_testing('../svm/doc_result/doc_merge_vec_test_200t.txt', _knn_test, train_class, K_NUM)
		print "Testing: " + timer.getTime()

		#Analysis result
		result = np.full(shape = (10, 10), fill_value = 0, dtype = 'int')
		for t, p in zip(target_out, predict_out):
			result[t, p] += 1
		
		
		correct_num = 0
		for i in range(10):
			correct_num += result[i][i]
		
		print "K= %d F1= %f"%(K_NUM, float(correct_num) / np.asarray(result).sum())
		print "Print Contigency Table"
		for i in range(len(result)):
			print ' '.join([str(x) for x in result[i]])
		
