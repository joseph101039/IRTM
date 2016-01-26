#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm
import tim

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


doc_result_loc = 'doc_result/'
doc_vec_loc = '/media/joseph/SSD/TrainingData_docVec_jieba/'

train_file = ['1t', '2t', '5t', '10t', '20t', '50t', '100t']
for tname in train_file:
	with open(doc_result_loc + "doc_merge_vec_train_" + tname + ".txt", 'r') as dr:
		target_class = dr.read().splitlines()
	target_class = [x.split(' ') for x in target_class]
	train_doc_list = [x.pop(0) for x in target_class]

	print 'doc_merge_vec_train_' + tname + '.txt'
	### prepare training target classes ###
	superClass = ["運動","娛樂","生活","社會","地方","要聞","兩岸","全球","產經","股市", "即時"]
	subClass = ["運動","娛樂","生活","社會","地方","要聞","兩岸","國際","財經"]
	target_out = []
	sup_in, sub_in = [-1, -1]
	for i in range(len(train_doc_list)):
		sup_index = superClass.index(target_class[i][0])
		if sup_index == 10: 
			sub_index = subClass.index(target_class[i][1])
			target_out.append(sub_index + 1)
		else:
			target_out.append(sup_index + 1)

	target_out = np.asarray(target_out)
	target_class = None



	#prepare training vectors
	train_vec = []
	for dname in train_doc_list:
		with open(doc_vec_loc + dname, 'r') as dr:
			vec = dr.read().splitlines()
			vec = [x.split(' ') for x in vec][0]
			vec = [float(k) for k in vec]
			train_vec.append(vec)

	train_vec = np.asarray(train_vec)

	print "Training Data Opened"
	t = Timer()
	### training by SVM classifier ##
	## choose a classifier ##
	clf = svm.SVC(probability = False, kernel = 'rbf')		#clf.decision_function_shape = "ovo"
	clf = svm.LinearSVC()	#clf.decision_function_shape = "ovr"

	clf.fit(train_vec, target_out)
	train_vec = None
	print "Training: " + t.getTimeGap()	
	#_____________________________________________________________#
	### testing ###
	with open(doc_result_loc + "doc_merge_vec_test_200t.txt", 'r') as dr:
		target_class = dr.read().splitlines()
	target_class = [x.split(' ') for x in target_class]
	test_doc_list = [x.pop(0) for x in target_class]

	#prepare testing target classes
	superClass = ["運動","娛樂","生活","社會","地方","要聞","兩岸","全球","產經","股市", "即時"]
	subClass = ["運動","娛樂","生活","社會","地方","要聞","兩岸","國際","財經"]
	target_out = []
	sup_in, sub_in = [-1, -1]
	for i in range(len(test_doc_list)):
		sup_index = superClass.index(target_class[i][0])
		if sup_index == 10: 
			sub_index = subClass.index(target_class[i][1])
			target_out.append(sub_index + 1)
		else:
			target_out.append(sup_index + 1)

	target_out = np.asarray(target_out)
	target_class = None


	#prepare testing vectors
	test_vec = []
	for dname in test_doc_list:
		with open(doc_vec_loc + dname, 'r') as dr:
			vec = dr.read().splitlines()
			vec = [x.split(' ') for x in vec][0]
			vec = [float(k) for k in vec]
			test_vec.append(vec)

	test_vec = np.asarray(test_vec)

	#print "Testing Data Opened"	###
	t.getTimeGap()
	
	### predict testing data probability ###
	#test_predict_proba(clf, test_vec, target_out)
	### predict classes ###

	ans = clf.predict(test_vec)
	print "Testing: " + t.getTimeGap()

	result = np.full(shape = (10, 10), fill_value = 0, dtype = 'int')
	for k in range(len(ans)):
		result[target_out[k] - 1, ans[k] -1] += 1
	
	correct_num = 0
	for i in range(10):
		correct_num += result[i][i]
	print "F1= %f" %(float(correct_num) / len(ans))
	
	print "Print Contingency Table"
	result = np.asarray(result)
	for i in range(len(result)):
		print ' '.join([str(x) for x in result[i]])


	

