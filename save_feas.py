from __future__ import print_function

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.ndimage
# from Net import Generator, WeightNet
from scipy.misc import imread, imsave
from skimage import transform, data
from glob import glob
import matplotlib.image as mpimg
import scipy.io as scio
import cv2
from pnet import PNet

from spat_ED import ED1
from spec_ED import ED2
from P2MSnet import pMS_ED
from MS2Pnet import pP_ED

from tensorflow.python import pywrap_tensorflow


MODEL_SAVE_PATH = './models/best_25-800-0.2/2000.ckpt'
MS2P_MODEL_SAVEPATH = './MS2P_models/2000/2000.ckpt'
P2MS_MODEL_SAVEPATH = './P2MS_models/2000/2000.ckpt'
SPAT_MODEL_SAVEPATH = './spat_models/2000/2000.ckpt'
SPEC_MODEL_SAVEPATH = './spec_models/2000/2000.ckpt'

path1 = 'test_imgs/pan/'
path2 = 'test_imgs/ms/'
output_path = 'features/'


def main():
	# print('\nBegin to generate pictures ...\n')
	"save features for examples"
	for i in range(50):
		file_name1 = path1 + str(i + 1) + '.png'
		file_name2 = path2 + str(i + 1) + '.tif'

		pan = imread(file_name1) / 255.0
		ms = imread(file_name2) / 255.0
		print('file1:', file_name1, 'shape:', pan.shape)
		print('file2:', file_name2, 'shape:', ms.shape)
		h1, w1 = pan.shape
		h2, w2, c = ms.shape



	# 	cpan = W0[0] * cv2.resize(ms[:, :, 0], (h1, w1)) + W0[1] * cv2.resize(ms[:, :, 1], (h1, w1)) + W0[
	# 		2] * cv2.resize(ms[:, :, 2], (h1, w1)) + W0[3] * cv2.resize(ms[:, :, 3], (h1, w1))
	# 	down_pan = np.expand_dims(cv2.resize(pan, (h2, w2)), axis = -1)
	# 	cms = np.concatenate([down_pan, down_pan, down_pan], axis = -1)
	# 	imsave('25_cpan.png', cpan)
	# 	imsave('25_cms.tif', cms)

		# # img1_Y = transform.resize(img1_Y, (h1, w1))
		# # img2_Y = transform.resize(img2_Y, (h1, w1))
		# h1, w1 = pan.shape

		#
		#
		# checkpoint_path = MODEL_SAVE_PATH
		# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
		# var_to_shape_map = reader.get_variable_to_shape_map()
		# for key in var_to_shape_map:
		# 	print("tensor_name: ", key)
		# 	print(reader.get_tensor(key).shape)
		#
	# with tf.Graph().as_default(), tf.Session() as sess:
	# 	MS = tf.placeholder(tf.float32, shape = (1, h2, w2, 4), name = 'MS')
	# 	PAN = tf.placeholder(tf.float32, shape = (1, h1, w1, 1), name = 'PAN')
	# 	Pnet = PNet('pnet')
	# 	X = Pnet.transform(PAN = PAN, ms = MS)
	#
	# 	with tf.device('/gpu:1'):
	# 		NET1 = ED1('spatial_ED')
	# 		_, SPAF1 = NET1.transform(PAN, is_training = False, reuse = False)
	# 	_, SPAF2 = NET1.transform(cPAN, is_training = False, reuse = True)
	# 	t_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'spatial_ED')
	#
	# 	NET2 = ED2('spectral_ED')
	# 	_, SPEF1 = NET2.transform(MS, is_training = False, reuse = False)
	# 	_, SPEF2 = NET2.transform(cMS, is_training = False, reuse = True)
	# 	t_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'spectral_ED')

		#
		#
		#
		# 	t_list = tf.trainable_variables()
		# # 	# for v in g_list:
		# # 	# 	if v not in t_list:
		# # 	# 		if ('beta' in v.name) or ('gamma' in v.name):
		# # 	# 			t_list.append(v)
		# #
		# 	saver = tf.train.Saver(var_list = t_list)
		# 	sess.run(tf.global_variables_initializer())
		# 	saver.restore(sess, MODEL_SAVE_PATH)
		# #
		# 	output = sess.run(X, feed_dict = {PAN: pan, MS: ms})
		#
		#

			
			# 	if c == 0:
			# 		cpan = W[c] * np.expand_dims(ms_us, axis = -1)
			# 	else:
			# 		cpan = cpan + W[c] * np.expand_dims(ms_us, axis = -1)
			# cpan = cpan.reshape([1, h1, w1, 1])
			# print('cpan shape:', cpan.shape)

		with tf.Graph().as_default(), tf.Session() as sess:
			INPUT1 = tf.placeholder(tf.float32, shape = (1, h1, w1, 1), name = 'INPUT')
			with tf.device('/gpu:0'):
				spatnet = ED1('spatial_ED')
				OUTPUT1 = spatnet.transform(INPUT1, is_training = False, reuse = False)
			spat_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'spatial_ED')

			MS = tf.placeholder(tf.float32, shape = (1, h1, w1, 4), name = 'MS')
			with tf.device('/gpu:1'):
				pPnet = pP_ED('pP_ED')
				MS_converted_PAN = pPnet.transform(I = MS, is_training = False, reuse = True)
			pP_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'pP_ED')

			t_list = tf.trainable_variables()
			sess.run(tf.global_variables_initializer())
			saver1 = tf.train.Saver(var_list = spat_list)
			saver1.restore(sess, SPAT_MODEL_SAVEPATH)
			saver2 = tf.train.Saver(var_list = pP_list)
			saver2.restore(sess, MS2P_MODEL_SAVEPATH)

			for c in range(4):
				if c==0:
					ms_us = cv2.resize(ms[:, :, c], (h1, w1))
					ms_us = ms_us.reshape([1, h1, w1, 1])
				else:
					ms_upsampled = cv2.resize(ms[:, :, c], (h1, w1))
					ms_upsampled = ms_upsampled.reshape([1, h1, w1, 1])
					ms_us = np.concatenate([ms_us, ms_upsampled], axis=-1)
			cpan = sess.run(MS_converted_PAN, feed_dict={MS: ms_us})
			pan = pan.reshape([1, h1, w1,1])
			spat_features1 = sess.run(spatnet.features, feed_dict = {INPUT1: pan})
			spat_features2 = sess.run(spatnet.features, feed_dict = {INPUT1: cpan})

			diff = np.mean(np.abs(spat_features1 - spat_features2), axis = (1, 2))
			print("diff shape:", diff.shape)
			if i == 0:
				Diff = diff
			else:
				Diff = np.concatenate([Diff, diff], axis = 0)

		scio.savemat('spat_diff.mat', {'D': Diff})


			# PAN = tf.placeholder(tf.float32, shape = (1, h1, w1, 1), name = 'MS')
			# with tf.device('/gpu:1'):
			# 	pMS_NET = pMS_ED('pMS_ED')
			# 	PAN_converted_MS = pMS_NET.transform(I = PAN, is_training = False, reuse = True)
			# pMS_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'pMS_ED')












if __name__ == '__main__':
	main()