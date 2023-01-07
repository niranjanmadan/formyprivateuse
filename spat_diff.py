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
from M2Cnet import pC_ED
from C2Mnet import pM_ED

from tensorflow.python import pywrap_tensorflow

C2M_MODEL_SAVEPATH = './C2M_models/6700/6700.ckpt'
M2C_MODEL_SAVEPATH = './M2C_models/6700/6700.ckpt'
SPAT_MODEL_SAVEPATH = './spat_models/8430/8430.ckpt'
SPEC_MODEL_SAVEPATH = './spec_models/8430/8430.ckpt'

path1 = 'test_imgs/ct/'
path2 = 'test_imgs/mri/'
output_path = 'features/'


def main():
	# print('\nBegin to generate pictures ...\n')
	"save features for examples"

	for root, dirs, files in os.walk(path1):
		print('files:', files)
	for i in range(len(files)):
		file_name1 = path1 + files[i]
		file_name2 = path2 + files[i]
		ct = imread(file_name1) / 255.0
		mri = imread(file_name2) / 255.0
		print('file1:', file_name1, 'shape:', ct.shape)
		print('file2:', file_name2, 'shape:', mri.shape)
		print("ct shape:", ct.shape)
		h1, w1 = ct.shape
		h2, w2 = mri.shape

		with tf.Graph().as_default(), tf.Session() as sess:
			INPUT = tf.placeholder(tf.float32, shape = (1, h1, w1, 1), name = 'INPUT')
			with tf.device('/gpu:0'):
				spatnet = ED1('spatial_ED')
				OUTPUT = spatnet.transform(INPUT, is_training = False, reuse = False)
			spat_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'spatial_ED')

			CT = tf.placeholder(tf.float32, shape = (1, h1, w1, 1), name = 'CT')
			with tf.device('/gpu:1'):
				pMnet = pM_ED('pM_ED')
				CT_converted_MRI = pMnet.transform(I = CT, is_training = False, reuse = False)
			pM_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'pM_ED')

			t_list = tf.trainable_variables()
			sess.run(tf.global_variables_initializer())
			saver1 = tf.train.Saver(var_list = spat_list)
			saver1.restore(sess, SPAT_MODEL_SAVEPATH)
			saver2 = tf.train.Saver(var_list = pM_list)
			saver2.restore(sess, C2M_MODEL_SAVEPATH)

			ct = ct.reshape([1, h1, w1, 1])
			ct_2_mri = sess.run(CT_converted_MRI, feed_dict = {CT: ct})
			mri = mri.reshape([1, h1, w1, 1])

			spat_features1 = sess.run(spatnet.features, feed_dict = {INPUT: mri})
			spat_features2 = sess.run(spatnet.features, feed_dict = {INPUT: ct_2_mri})

			diff = np.mean(np.abs(spat_features1 - spat_features2), axis = (1, 2))
			if i == 0:
				Diff = diff
			else:
				Diff = np.concatenate([Diff, diff], axis = 0)

	Diff = np.mean(Diff, axis=0)
	channel_sort = np.flip(np.argsort(Diff), axis=0)
	sorted_Diff = sorted(Diff, reverse=True)

	f = "spat_diff.txt"
	for i in range(len(channel_sort)):
		if i==0:
			with open(f, "w") as file:
				file.write(str(channel_sort[i]) + "\n")
		else:
			with open(f, "a") as file:
				file.write(str(channel_sort[i]) + "\n")
	# scio.savemat('spat_diff.mat', {'D': Diff})



if __name__ == '__main__':
	main()