from __future__ import print_function

import time
import os
import numpy as np
import tensorflow as tf
import scipy.ndimage
from scipy.misc import imread, imsave
from skimage import transform, data
from glob import glob
import scipy.io as scio
import cv2
from fnet import FNet
import matplotlib.pyplot as plt
import time

from tensorflow.python import pywrap_tensorflow
from tqdm import tqdm
from scipy.misc import imread, imsave

MODEL_SAVE_PATH = './models/3200/3200.ckpt'
path1 = 'test_imgs/ct/'
path2 = 'test_imgs/mri/'
output_path = 'results/'

def main():
	print('\nBegin to generate pictures ...\n')
	t=[]
	for root, dirs, files in os.walk(path1):
		print('files:', files)

	with tf.Graph().as_default() as graph:
		with tf.Session() as sess:
			CT = tf.placeholder(tf.float32, shape=(1, None, None, 1), name='CT')
			MRI = tf.placeholder(tf.float32, shape=(1, None, None, 1), name='MRI')
			Fnet = FNet('fnet')
			X = Fnet.transform(CT=CT, MRI=MRI)

			t_list = tf.trainable_variables()
			g_list = tf.global_variables()

			saver = tf.train.Saver(var_list=t_list)
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, MODEL_SAVE_PATH)

			for i in range(len(files)):
				begin = time.time()
				file_name1 = path1 + files[i]
				file_name2 = path2 + files[i]
				ct = imread(file_name1) / 255.0
				mri = imread(file_name2) / 255.0
				print('file1:', file_name1, 'shape:', ct.shape)
				print('file2:', file_name2, 'shape:', mri.shape)
				h1, w1 = ct.shape
				h2, w2 = mri.shape
				ct = ct.reshape([1, h1, w1, 1])
				mri = mri.reshape([1, h1, w1, 1])

				output = sess.run(X, feed_dict = {CT: ct, MRI: mri})
				imsave(output_path + file_name1.split('/')[-1], output[0, :, :, 0])
				end = time.time()
				t.append(end - begin)
				print("Time: %s" % (end-begin))
		print("Time: mean: %s,, std: %s" % (np.mean(t), np.std(t)))
		scio.savemat('time.mat', {'time': t})

if __name__ == '__main__':
	main()
