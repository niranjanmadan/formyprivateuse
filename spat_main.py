from __future__ import print_function
import time
import os
import h5py
import numpy as np
import scipy.ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import cv2
from spat_ED import ED1
from C2Mnet import pM_ED

path = 'CT-MRI_114.h5'

EPOCHES = 10
BATCH_SIZE = 16
patch_size = 114
logging_period = 10
LEARNING_RATE = 0.001
DECAY_RATE = 0.85

# W = [0.31, 0.05, 0.37, 0.27]
# W = [0.35, 0.05, 0.35, 0.25]

C2M_MODEL_PATH = './C2M_models/6700/6700.ckpt'

def main():
	with tf.device('/cpu:0'):
		source_data = h5py.File(path, 'r')
		source_data = source_data['data'][:]
		data = np.transpose(source_data, (0, 3, 2, 1)) / 255.0
		print("source_data shape:", data.shape)

		# ms_data = np.zeros(shape=(99, 99, 4), dtype=np.float32)
		# for i in range(20):
		# 	fig = plt.figure()
		# 	f1 = fig.add_subplot(311)
		# 	f2 = fig.add_subplot(312)
		# 	f3 = fig.add_subplot(313)
		# 	for c in range(4):
		# 		# ms_data[:, :, c] = scipy.ndimage.zoom(gt_data[i, :, :, c], 0.25, order=1)
		# 		ms_data[:, :, c] = cv2.resize(data[i, :, :, c], (99, 99))
		#
		# 	f1.imshow(ms_data[:, :, 0:3])
		# 	f2.imshow(data[i, :, :, 4], cmap = 'gray')
		# 	f3.imshow(data[i, :, :, 0:3])
		# 	plt.show()

		start_time = datetime.now()
		print('Epoches: %d, Batch_size: %d' % (EPOCHES, BATCH_SIZE))

		num_imgs = data.shape[0]
		mod = num_imgs % BATCH_SIZE
		n_batches = int(num_imgs // BATCH_SIZE)
		print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))
		if mod > 0:
			print('Train set has been trimmed %d samples...\n' % mod)
			source_imgs = data[:-mod]
		else:
			source_imgs = data

		# create the graph
		with tf.Graph().as_default(), tf.Session() as sess:
			CT = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'CT')
			with tf.device('/gpu:0'):
				pM_NET = pM_ED('pM_ED')
				CT_converted_MRI = pM_NET.transform(I = CT, is_training = True, reuse = False)

			pM_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'pM_ED')


			with tf.device('/gpu:0'):
				INPUT = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'INPUT')
				NET=ED1('spatial_ED')
				OUTPUT = NET.transform(INPUT, is_training=True, reuse=False)

			LOSS = 40 * tf.reduce_mean(tf.square(OUTPUT - INPUT)) + (1 - tf.reduce_mean(SSIM_LOSS(OUTPUT, INPUT), axis=0))

			current_iter = tf.Variable(0)
			learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE, global_step = current_iter,
			                                           decay_steps = int(n_batches), decay_rate = DECAY_RATE,
			                                           staircase = False)

			theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'spatial_ED')
			solver = tf.train.AdamOptimizer(learning_rate).minimize(LOSS, global_step = current_iter, var_list = theta)
			

			sess.run(tf.global_variables_initializer())
			saver0 = tf.train.Saver(var_list = pM_list)
			saver0.restore(sess, C2M_MODEL_PATH)
			saver = tf.train.Saver(max_to_keep = 20)
			tf.summary.scalar('Loss', LOSS)
			tf.summary.scalar('Learning rate', learning_rate)
			tf.summary.image('input', INPUT, max_outputs = 3)
			tf.summary.image('output', OUTPUT, max_outputs = 3)

			merged = tf.summary.merge_all()
			writer = tf.summary.FileWriter("spat_logs/", sess.graph)

			# ** Start Training **
			step = 0


			for epoch in range(EPOCHES):
				np.random.shuffle(source_imgs)
				for batch in range(n_batches):
					step += 1
					current_iter = step
					ct_batch = data[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
					mri_batch = data[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
					ct_batch = np.expand_dims(ct_batch, -1)
					mri_batch = np.expand_dims(mri_batch, -1)
					# for c in range(4):
					# 	if c == 0:
					# 		ms_c_pan_batch = W[c] * np.expand_dims(ms_batch[:, :, :, c], axis = -1)
					# 	else:
					# 		ms_c_pan_batch = ms_c_pan_batch + W[c] * np.expand_dims(ms_batch[:, :, :, c], axis = -1)
					ct_2_ms_batch = sess.run(CT_converted_MRI, feed_dict={CT: ct_batch})


					# run the training step
					if step % 2:
						FEED_DICT = {INPUT: mri_batch}
						sess.run(solver, feed_dict = FEED_DICT)
					else:
						FEED_DICT = {INPUT: ct_2_ms_batch}
						sess.run(solver, feed_dict = FEED_DICT)
					result = sess.run(merged, feed_dict = FEED_DICT)
					writer.add_summary(result, step)
					if step % 100 == 0:
						saver.save(sess, 'spat_models/' + str(step) + '/' + str(step) + '.ckpt')

					is_last_step = (epoch == EPOCHES - 1) and (batch == n_batches - 1)
					if is_last_step or step % logging_period == 0:
						elapsed_time = datetime.now() - start_time
						loss = sess.run(LOSS, feed_dict = FEED_DICT)
						lr = sess.run(learning_rate)
						print('Epoch:%d/%d: step:%d, lr:%s, loss:%s, elapsed_time:%s\n' % (
							epoch + 1, EPOCHES, step, lr, loss, elapsed_time))
				saver.save(sess, 'spat_models/' + str(step) + '/' + str(step) + '.ckpt')

		writer.close()


def SSIM_LOSS(img1, img2, size = 11, sigma = 1.5):
	window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
	k1 = 0.01
	k2 = 0.03
	L = 1  # depth of image (255 in case the image has a different scale)
	c1 = (k1 * L) ** 2
	c2 = (k2 * L) ** 2
	mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
	mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')
	mu1_sq = mu1 * mu1
	mu2_sq = mu2 * mu2
	mu1_mu2 = mu1 * mu2
	sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_sq
	sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
	sigma1_2 = tf.nn.conv2d(img1 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2

	# value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
	ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
	value = tf.reduce_mean(ssim_map, axis = [1, 2, 3])
	return value


def _tf_fspecial_gauss(size, sigma):
	"""Function to mimic the 'fspecial' gaussian MATLAB function
	"""
	x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

	x_data = np.expand_dims(x_data, axis = -1)
	x_data = np.expand_dims(x_data, axis = -1)

	y_data = np.expand_dims(y_data, axis = -1)
	y_data = np.expand_dims(y_data, axis = -1)

	x = tf.constant(x_data, dtype = tf.float32)
	y = tf.constant(y_data, dtype = tf.float32)

	g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
	return g / tf.reduce_sum(g)



if __name__ == '__main__':
	main()
