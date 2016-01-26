from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
import numpy as np
from dnn_get_input import *
import keras
#########   Function ######### 

def softmax(x):
	exp_x = np.exp(x)
	exp_sum = np.sum(exp_x, axis=1)
	return exp_x/exp_sum[:,None]

#########   Input Training Data ######### 

#########   Testing Data #########  
model = Sequential()
INPUT_DIM = 300
OUTPUT_DIM = 10

model.add(Dense(1500, input_dim=INPUT_DIM, init='uniform'))	#original para: activation='relu'
model.add(PReLU(init='zero', weights=None))
model.add(Dropout(0.2))

model.add(Dense(3000, init='uniform'))
model.add(PReLU(init='zero', weights=None))
model.add(Dropout(0.5))

model.add(Dense(1000, init='uniform'))
model.add(PReLU(init='zero', weights=None))
model.add(Dropout(0.3))

model.add(Dense(500, init='uniform'))
model.add(PReLU(init='zero', weights=None))
model.add(Dropout(0.1))

model.add(Dense(100, init='uniform'))
model.add(PReLU(init='zero', weights=None))
#model.add(Dropout(0.1))

model.add(Dense(OUTPUT_DIM, init='uniform'))
model.add(Activation('softmax'))

adagrad = keras.optimizers.Adagrad(lr=0.04, epsilon=1e-04)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adagrad', class_mode='categorical')	


#data preprocessing for training data
docList, targ_out = openTrainResult('doc_merge_vec_train_2t.txt')
train_vec = openTrainArk(docList)
targ_vec = []
for i in range(len(targ_out)):
	vec = np.full(shape = 10, fill_value = 0, dtype='float32')
	vec[targ_out[i]] = 1
	targ_vec.append(vec)

train_vec = np.asarray(train_vec, dtype = 'float32')
targ_vec = np.asarray(targ_vec, dtype = 'float32')

#data preprocessing for validation data
test_docList, test_out = openTrainResult('doc_merge_vec_test_200t.txt')
test_vec_valid = openTrainArk(test_docList[:5000])
targ_vec_valid = []
for i in range(5000):
	vec = np.full(shape = 10, fill_value = 0, dtype='float32')
	vec[test_out[i]] = 1
	targ_vec_valid.append(vec)
test_vec_valid = np.asarray(test_vec_valid, dtype = 'float32')
targ_vec_valid = np.asarray(targ_vec_valid, dtype = 'float32')

t = Timer()
model.fit(train_vec, targ_vec, nb_epoch=10, batch_size=100, show_accuracy=True, shuffle=True, verbose=1, validation_data = (test_vec_valid, targ_vec_valid), callbacks = [early_stop])
print "Training: " + t.getTimeGap()


#data preprocessing for testing data
train_vec = None
targ_vec = None
test_vec_valid = None
targ_vec_valid = None

test_vec = openTrainArk(test_docList)
targ_vec = []
for i in range(len(test_out)):
	vec = np.full(shape = 10, fill_value = 0, dtype='float32')
	vec[test_out[i]] = 1
	targ_vec.append(vec)

test_vec = np.asarray(test_vec, dtype = 'float32')
targ_vec = np.asarray(targ_vec, dtype = 'float32')

t.getTimeGap()
score = model.evaluate(test_vec, targ_vec, show_accuracy=True, verbose=1)
print "Testing: " + t.getTimeGap()
print score	#score = [loss, F1]
