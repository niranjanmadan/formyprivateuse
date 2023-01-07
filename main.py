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
from fnet import FNet
from spat_ED import ED1
from spec_ED import ED2

from M2Cnet import pC_ED
from C2Mnet import pM_ED

from VGGnet.vgg16 import Vgg16

# from tensorflow.python import pywrap_tensorflow

path = 'CT-MRI_22.h5'

EPOCHES = 4
BATCH_SIZE = 32
patch_size = 22
logging_period = 10
LEARNING_RATE = 0.0005
DECAY_RATE = 0.7
c = 0.5 # 30000

C2M_MODEL_SAVEPATH = './C2M_models/6700/6700.ckpt'
M2C_MODEL_SAVEPATH = './M2C_models/6700/6700.ckpt'
SPAT_MODEL_SAVEPATH = './spat_models/8430/8430.ckpt'
SPEC_MODEL_SAVEPATH = './spec_models/8430/8430.ckpt'


SPAT_INDEX = np.loadtxt("spat_diff.txt", dtype = np.int32)
# print("SPAT_INDEX:", SPAT_INDEX)
SPEC_INDEX = np.loadtxt("spec_diff.txt", dtype = np.int32)
# print("SPEC_INDEX:", SPEC_INDEX)
FEA_NUM = 10


def main():
	with tf.device('/cpu:0'):
		source_data = h5py.File(path, 'r')
		data = source_data['data'][:]
		data = np.transpose(data, (0, 3, 2, 1)) / 255.0
		print("data max: %s, min: %s" % (np.max(data), np.min(data)))
		# shape = pan_data.shape
		# N=int(shape[0])

		# data = np.concatenate([gt_data, pan_data], axis = -1)
		print("data shape:", data.shape)

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
		MRI = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'MRI')
		W_CT = tf.placeholder(tf.float32, shape = (BATCH_SIZE), name = 'W_CT')
		W_MRI = tf.placeholder(tf.float32, shape = (BATCH_SIZE), name = 'W_MRI')

		with tf.device('/gpu:0'):
			Fnet = FNet('fnet')
			X = Fnet.transform(CT = CT, MRI = MRI)
		theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'fnet')

		# grad_CT = grad(CT)
		# grad_MRI = grad(MRI)

		# surface loss
		''' SSIM loss'''
		SSIM1 = 1 - SSIM(CT, X) #
		SSIM2 = 1 - SSIM(MRI, X) #  #1 - SSIM_LOSS(CT, X)
		mse1 = Fro_LOSS(X - CT)
		mse2 = Fro_LOSS(X - MRI)

		LOSS_SSIM = tf.reduce_mean(W_CT * SSIM1 + W_MRI * SSIM2)  #tf.reduce_mean( W1* SSIM1 + W2 * SSIM2)
		LOSS_MSE = tf.reduce_mean(W_CT * mse1 + W_MRI * mse2)
		LOSS_TEXTURE = LOSS_SSIM + 10 * LOSS_MSE


		"feature loss"
		"spatial"
		with tf.device('/gpu:1'):
			NET1 = ED1('spatial_ED')
			_ = NET1.transform(MRI, is_training = False, reuse = False)
			SPAF1= NET1.features
			_ = NET1.transform(X, is_training = False, reuse = True)
			SPAF2= NET1.features
		spat_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'spatial_ED')

		with tf.device('/cpu:0'):
			for num in range(FEA_NUM):
				index = SPAT_INDEX[num]
				print(index)
				if num == 0:
					LOSS_spat_fea = tf.reduce_mean(tf.abs(SPAF1[:, :, :, index] - SPAF2[:, :, :, index]))
				else:
					LOSS_spat_fea = LOSS_spat_fea + tf.reduce_mean(
						tf.square(SPAF1[:, :, :, index] - SPAF2[:, :, :, index]))
			LOSS_spat_fea = LOSS_spat_fea / FEA_NUM

		"spectral"
		with tf.device('/gpu:1'):
			NET2 = ED2('spectral_ED')
			_ = NET2.transform(CT, is_training = False, reuse = False)
			SPEF1 = NET2.features
			_ = NET2.transform(X, is_training = False, reuse = True)
			SPEF2 = NET2.features
		spec_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'spectral_ED')

		with tf.device('/cpu:0'):
			for num in range(FEA_NUM):
				index = SPEC_INDEX[num]
				print(index)
				if num == 0:
					LOSS_spec_fea = tf.reduce_mean(tf.square(SPEF1[:, :, :, index] - SPEF2[:, :, :, index]))
				else:
					LOSS_spec_fea = LOSS_spec_fea + tf.reduce_mean(tf.abs(SPEF1[:, :, :, index] - SPEF2[:, :, :, index]))
			LOSS_spec_fea = LOSS_spec_fea / FEA_NUM
			LOSS_UNIQUE = 1 * LOSS_spec_fea + 1 * LOSS_spat_fea # 800 * (LOSS_spat_fea + 0.2 * LOSS_spec_fea)

		LOSS = LOSS_TEXTURE + 0.5 * LOSS_UNIQUE

		current_iter = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE, global_step = current_iter,
		                                           decay_steps = int(n_batches), decay_rate = DECAY_RATE,
		                                           staircase = False)
		# solver = tf.train.RMSPropOptimizer(learning_rate).minimize(LOSS, global_step = current_iter, var_list = theta)
		solver = tf.train.AdamOptimizer(learning_rate).minimize(LOSS, global_step = current_iter,
		                                                           var_list = theta)

		# sess.run(tf.global_variables_initializer())
		sess.run(tf.global_variables_initializer())
		# saver0 = tf.train.Saver(var_list = theta)
		saver1 = tf.train.Saver(var_list = spat_list)
		saver2 = tf.train.Saver(var_list = spec_list)
		# saver0.restore(sess, MS2P_MODEL_SAVEPATH)
		saver1.restore(sess, SPAT_MODEL_SAVEPATH)
		saver2.restore(sess, SPEC_MODEL_SAVEPATH)

		saver = tf.train.Saver(max_to_keep = 10)
		tf.summary.scalar('Loss_texture', LOSS_TEXTURE)
		tf.summary.scalar('Loss_ssim', LOSS_SSIM)
		tf.summary.scalar('Loss_mse', LOSS_MSE)
		tf.summary.scalar('Loss_unique', LOSS_UNIQUE)
		tf.summary.scalar('Loss', LOSS)

		tf.summary.image('MRI', MRI, max_outputs = 3)
		tf.summary.image('CT', CT, max_outputs = 3)
		tf.summary.image('X', X, max_outputs = 3)


		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("logs/", sess.graph)

		# ** Start Training **
		step = 0
		for epoch in range(EPOCHES):
			np.random.shuffle(source_imgs)
			for batch in range(n_batches):
				step += 1
				current_iter = step
				ct_batch = np.expand_dims(data[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0], axis = -1)
				mri_batch = np.expand_dims(data[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1], axis = -1)
				# run the training step

				# with tf.device('/gpu:1'):
				# 	S1_VGG_in = tf.image.resize_nearest_neighbor(CT, size = [224, 224])
				# 	S1_VGG_in = tf.concat((S1_VGG_in, S1_VGG_in, S1_VGG_in), axis = -1)
				# 	S2_VGG_in = tf.image.resize_nearest_neighbor(MRI, size = [224, 224])
				# 	S2_VGG_in = tf.concat((S2_VGG_in, S2_VGG_in, S2_VGG_in), axis = -1)
				#
				# 	S1_FEAS=[]
				# 	S2_FEAS=[]
				# 	vgg1 = Vgg16()
				# 	with tf.name_scope("vgg1"):
				# 		FEAS = vgg1.build(tf.concat([S1_VGG_in, S2_VGG_in], axis=0))
				#
				# 	for i in range(len(FEAS)):
				# 		temp = FEAS[i]
				# 		S1_FEAS.append(temp[0: BATCH_SIZE, :, :, :])
				# 		S2_FEAS.append(temp[BATCH_SIZE: 2 * BATCH_SIZE, :, :, :])
				#
				# 	for i in range(len(S1_FEAS)):
				# 		m1 = tf.reduce_mean(tf.square(features_grad(S1_FEAS[i])), axis = [1, 2, 3])
				# 		m2 = tf.reduce_mean(tf.square(features_grad(S2_FEAS[i])), axis = [1, 2, 3])
				# 		if i == 0:
				# 			ws1 = tf.expand_dims(m1, axis = -1)
				# 			ws2 = tf.expand_dims(m2, axis = -1)
				# 		else:
				# 			ws1 = tf.concat([ws1, tf.expand_dims(m1, axis = -1)], axis = -1)
				# 			ws2 = tf.concat([ws2, tf.expand_dims(m2, axis = -1)], axis = -1)
				#
				# s1 = tf.reduce_mean(ws1, axis = -1) / c
				# s2 = tf.reduce_mean(ws2, axis = -1) / c
				# s = tf.nn.softmax(tf.concat([tf.expand_dims(s1, axis = -1), tf.expand_dims(s2, axis = -1)], axis = -1))


				ct_en = EN(ct_batch)
				mri_en = EN(mri_batch)
				ct_intensity = intensity(ct_batch)# max(ct_batch) #
				mri_intensity = intensity(mri_batch)# max(mri_batch) #

				s_ct = ct_en + 3 * ct_intensity
				s_mri = mri_en + 3 * mri_intensity
				w_ct = np.exp(s_ct / c) / (np.exp(s_ct / c) + np.exp(s_mri / c))
				w_mri = np.exp(s_mri / c) / (np.exp(s_ct / c) + np.exp(s_mri / c))

				FEED_DICT = {CT: ct_batch, MRI: mri_batch, W_CT: w_ct, W_MRI: w_mri}
				sess.run(solver, feed_dict = FEED_DICT)

				result = sess.run(merged, feed_dict = FEED_DICT)
				writer.add_summary(result, step)
				if step % 200 == 0:
					saver.save(sess, 'models/' + str(step) + '/' + str(step) + '.ckpt')

				is_last_step = (epoch == EPOCHES - 1) and (batch == n_batches - 1)
				if is_last_step or step % logging_period == 0:
					elapsed_time = datetime.now() - start_time
					loss = sess.run(LOSS, feed_dict = FEED_DICT)
					lr = sess.run(learning_rate)
					print('Epoch: %d/%d, Step: %d/%d, Loss: %s, Lr: %s, Time: %s\n' % (
						epoch + 1, EPOCHES, step % n_batches, n_batches, loss, lr, elapsed_time))

				# w1, w2 = sess.run([W_CT, W_MRI], feed_dict = FEED_DICT)
				# print("w1: %s, w2: %s" % (w1[0], w2[0]))
				# print("w1: %s, w2: %s" % (w1[1], w2[1]))
				# print("w1: %s, w2: %s\n" % (w1[2], w2[2]))
				# fig = plt.figure()
				# f1 = fig.add_subplot(321)
				# f2 = fig.add_subplot(322)
				# f3 = fig.add_subplot(323)
				# f4 = fig.add_subplot(324)
				# f5 = fig.add_subplot(325)
				# f6 = fig.add_subplot(326)
				# f1.imshow(ct_batch[0, :, :, 0], cmap='gray')
				# f2.imshow(mri_batch[0, :, :, 0], cmap='gray')
				# f3.imshow(ct_batch[1, :, :, 0], cmap='gray')
				# f4.imshow(mri_batch[1, :, :, 0], cmap='gray')
				# f5.imshow(ct_batch[2, :, :, 0], cmap='gray')
				# f6.imshow(mri_batch[2, :, :, 0], cmap='gray')
				# plt.show()




def SSIM(img1, img2, size = 11, sigma = 1.5):
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


def features_grad(features):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	_, _, _, c = features.shape
	c = int(c)
	for i in range(c):
		fg = tf.nn.conv2d(tf.expand_dims(features[:, :, :, i], axis = -1), kernel, strides = [1, 1, 1, 1],
		                  padding = 'SAME')
		if i == 0:
			fgs = fg
		else:
			fgs = tf.concat([fgs, fg], axis = -1)
	return fgs


def SSIM_LOSS(img1, img2, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    k1 = 0.01
    k2 = 0.03
    L = 1  # depth of image (255 in case the image has a different scale)
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma1_2 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2

    # value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    value = tf.reduce_mean(ssim_map, axis=[1, 2, 3])
    return value


def UIQI(img1, img2, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    k1 = 0.01
    k2 = 0.03
    L = 1  # depth of image (255 in case the image has a different scale)
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma1_2 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2

    # value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    #ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    uiqi_map = ((2 * sigma1_2 + c2) * (2 * mu1 * mu2 + c1))/((sigma1_sq + sigma2_sq+c1)*(tf.square(mu1)+tf.square(mu2)+c2))
    value = tf.reduce_mean(uiqi_map, axis=[1, 2, 3])
    return value


def Fro_LOSS(batchimg):
	fro_norm = tf.square(tf.norm(batchimg, axis = [1, 2], ord = 'fro')) / (int(batchimg.shape[1]) * int(batchimg.shape[2]))
	E = tf.reduce_mean(fro_norm, axis = -1)
	return E


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


def smooth(I):
	kernel = tf.constant([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	B, H, W, C= I.get_shape().as_list()
	for c in range(C):
		if c == 0:
			img = tf.nn.conv2d(tf.expand_dims(I[:, :, :, c], axis = -1), kernel, strides = [1, 1, 1, 1],
			                   padding = 'SAME')
		else:
			img = tf.concat([img, tf.nn.conv2d(tf.expand_dims(I[:, :, :, c], axis = -1), kernel, strides = [1, 1, 1, 1],
			                                   padding = 'SAME')], axis = -1)
	return img


def grad(I):
	kernel = tf.constant([[-1 / 8, -1 / 8, -1 / 8], [-1 / 8, 1, -1 / 8], [-1 / 8, -1 / 8, -1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	B, H, W, C= I.get_shape().as_list()
	for c in range(C):
		if c == 0:
			grad = tf.nn.conv2d(tf.expand_dims(I[:, :, :, c], axis = -1), kernel, strides = [1, 1, 1, 1],
			                   padding = 'SAME')
		else:
			grad = tf.concat([grad, tf.nn.conv2d(tf.expand_dims(I[:, :, :, c], axis = -1), kernel, strides = [1, 1, 1, 1],
			                                   padding = 'SAME')], axis = -1)
	return grad


def EN(inputs):
	len = inputs.shape[0]
	entropies = np.zeros(shape = (len))
	grey_level = 256
	counter = np.zeros(shape = (grey_level))

	for i in range(len):
		input_uint8 = (inputs[i, :, :, 0] * 255).astype(np.uint8)
		input_uint8 = input_uint8 + 1
		for m in range(patch_size):
			for n in range(patch_size):
				indexx = input_uint8[m, n]
				counter[indexx] = counter[indexx] + 1
		total = np.sum(counter)
		p = counter / total
		for k in range(grey_level):
			if p[k] != 0:
				entropies[i] = entropies[i] - p[k] * np.log2(p[k])
	return entropies


def intensity(inputs):
	len = inputs.shape[0]
	intensities = np.zeros(shape = (len))
	for i in range(len):
		input = inputs[i, :, :, 0]
		logic = input > 0.85
		input = input * logic
		input = input.reshape([-1])
		# exist = (input != 0)
		num = input.sum(axis = 0)
		# den = exist.sum(axis = 0)
		intensities[i] = num/(inputs.shape[1] * inputs.shape[2])# / den
	return intensities

def max(inputs):
	len = inputs.shape[0]
	intensities = np.zeros(shape = (len, 1))
	for i in range(len):
		input = inputs[i, :, :, 0]
		intensities[i, 0] = np.max(input)
	return intensities

if __name__ == '__main__':
	main()
