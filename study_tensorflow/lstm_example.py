# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:20:29 2017

@author: GM
"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

import h5py

import tflearn
import numpy as np
from tflearn.data_utils import image_preloader
from text2vector import *
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Building Residual Network
#trainX = pad_sequences(trainX, maxlen=100, value=0.)
def predict_text(X):
	# Network building

	#testX = get_record(audio)
	testX = pad_sequences(X, maxlen=40, value=0.)
	print("\033[94m {}\033[00m" .format("I'm thinking your emotion..."))
	#tf.reset_default_graph()	
	with tf.Graph().as_default():

	    # Network building
	    net = tflearn.input_data([None, 40])
	    # Masking is not required for embedding, sequence length is computed prior to
	    # the embedding op and assigned as 'seq_length' attribute to the returned Tensor.
	    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
	    #net = tflearn.lstm(net, 128, dropout=0.8, dynamic=True)
	    net = tflearn.lstm(net, 128, dropout=0.8)
	    net = tflearn.fully_connected(net, 2, activation='softmax')
	    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
	                             loss='categorical_crossentropy')

	    # Training
	    model = tflearn.DNN(net, checkpoint_path='dynamic/model_dynamic_fix', tensorboard_verbose=0)
	    model.load('dynamic/model_dynamic_fix-33750')

	    result = model.predict(testX)
	    print("\033[94m {}\033[00m" .format('predicted confidence:') + '    '  + str(result))

	#test_X, test_Y = image_preloader('/home/spc/xinyuan/test_data_1000', image_shape=(227, 227),   mode='folder', categorical_labels=True,   normalize=True)
	#train, vali, test = imdb.load_data(path='new_tweet.pkl', n_words=10000,
	#                                valid_portion=0.1)
	#testX, testY = test
	#testX = [[12, 345, 731],[12, 21, 27, 1697],[44, 359]]
	#testX = [[12, 345, 731]]

	# Converting labels to binary vectors
	#trainY = to_categorical(trainY, nb_classes=2)
	#testY = to_categorical(testY, nb_classes=2)

	#model.load('resnext_227.tfl')

	#print(confidence)
	#print(result.shape[0])

	for i in range(result.shape[0]):
		#print(result[i][0])
		if result[i][0] <= 0.1 and result[i][1] >= 0.9:
			print("\033[94m {}\033[00m" .format('I think your emotion is:') + '\033[1m'+ "\033[92m {}\033[00m" .format('positive'))
		elif result[i][0] >= 0.9 and result[i][1] <=0.1:
			print("\033[94m {}\033[00m" .format('I think your emotion is:') +'\033[1m'+ "\033[91m {}\033[00m" .format('negative'))
		else:
			print("\033[94m {}\033[00m" .format('I think your emotion is:') +'\033[1m'+ "\033[93m {}\033[00m" .format('nutral'))

if __name__ == '__main__':
	predict_text("samples/Buffer/buffer.wav")
#predict_y = model.predict(test_X)
#predicted_labels = np.argmax(predict_y,axis = 1)
#model.load('googlenet_3classes.tfl')
#model = tflearn.Evaluator(network)
#score = model.evaluate(testX,testY,batch_size=16)
##print('Test accuarcy: %0.4f%%' %  np.sum((test_Y == predicted_labels) / len(test_Y)))
#print(type(score))#

#k = len(score)
#print(k)
#for i in range(k):
#    print('Test accuarcy: %0.4f%%' %  (score[i]*100))


