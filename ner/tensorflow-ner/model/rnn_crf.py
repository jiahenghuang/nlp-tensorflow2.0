#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-28 上午10:20
# @Author  : 林利芳
# @File    : rnn_crf.py
import tensorflow as tf
from config.hyperparams import HyperParams as hp
from model.module.modules import embedding
from model.module.rnn import ForgetLSTMCell, IndRNNCell


class BiRnnCRF(object):
	def __init__(self, vocab_size, num_tags):
		self.vocab_size = vocab_size
		self.num_tags = num_tags
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.x = tf.placeholder(dtype=tf.int32, shape=[None, hp.max_len])
			self.y = tf.placeholder(dtype=tf.int32, shape=[None, hp.max_len])
			self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])
			self.dropout_keep_prob = tf.placeholder(tf.float32, name="keep_prob")
			self.global_step = tf.train.create_global_step()
			outputs = embedding(self.x, vocab_size=self.vocab_size, num_units=hp.num_units, scale=True, scope="embed")
			
			# cnn rnn 层
			
			outputs = self.rnn_layer(outputs)
			outputs = self.cnn_layer(outputs)
			self.logits = self.logits_layer(outputs)
			# crf 层
			self.loss, self.transition = self.crf_layer()
			# 优化器
			self.train_op = self.optimize()
	
	def rnn_layer(self, inputs, seg=hp.seg):
		"""
		创建双向RNN层
		:param inputs: 输入
		:param seg: LSTM GRU F-LSTM, IndRNN
		:return:
		"""
		if seg == 'LSTM':
			fw_cell = [tf.nn.rnn_cell.LSTMCell(num_units=hp.num_units) for _ in range(hp.num_layer)]
			bw_cell = [tf.nn.rnn_cell.LSTMCell(num_units=hp.num_units) for _ in range(hp.num_layer)]
		
		elif seg == 'GRU':
			fw_cell = [tf.nn.rnn_cell.GRUCell(num_units=hp.num_units) for _ in range(hp.num_layer)]
			bw_cell = [tf.nn.rnn_cell.GRUCell(num_units=hp.num_units) for _ in range(hp.num_layer)]
		elif seg == 'F-LSTM':
			fw_cell = [ForgetLSTMCell(num_units=hp.num_units) for _ in range(hp.num_layer)]
			bw_cell = [ForgetLSTMCell(num_units=hp.num_units) for _ in range(hp.num_layer)]
		elif seg == 'IndRNN':
			fw_cell = [IndRNNCell(num_units=hp.num_units) for _ in range(hp.num_layer)]
			bw_cell = [IndRNNCell(num_units=hp.num_units) for _ in range(hp.num_layer)]
		else:
			fw_cell = [tf.nn.rnn_cell.BasicRNNCell(num_units=hp.num_units) for _ in range(hp.num_layer)]
			bw_cell = [tf.nn.rnn_cell.BasicRNNCell(num_units=hp.num_units) for _ in range(hp.num_layer)]
		# 双向rnn
		outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
																	   sequence_length=self.seq_lens,
																	   dtype=tf.float32)
		# 合并双向rnn的output batch_size * max_seq * (hidden_dim*2)
		fw_output, bw_output = tf.split(outputs, 2, axis=-1)
		outputs = tf.add(fw_output, bw_output)
		print(outputs.get_shape().as_list())
		return outputs
	
	@staticmethod
	def cnn_layer(inputs):
		inputs = tf.expand_dims(inputs, axis=-1)
		outputs = []
		channel = hp.num_units // len(hp.filters)
		for ii, width in enumerate(hp.filters):
			with tf.variable_scope("cnn_{}_layer".format(ii)):
				weight = tf.Variable(tf.truncated_normal([width, hp.num_units, 1, channel], stddev=0.1, name='w'))
				bias = tf.get_variable('bias', [channel], initializer=tf.constant_initializer(0.0))
				output = tf.nn.conv2d(inputs, weight, strides=[1, 1, hp.num_units, 1], padding='SAME')
				output = tf.nn.bias_add(output, bias, data_format="NHWC")
				output = tf.nn.relu(output)
				output = tf.reshape(output, shape=[-1, hp.max_len, channel])
				outputs.append(output)
		outputs = tf.concat(outputs, axis=-1)
		return outputs
	
	def logits_layer(self, outputs):
		"""
		loggits
		:param outputs:
		:return:
		"""
		w = tf.get_variable(name='w', dtype=tf.float32, shape=[hp.num_units, self.num_tags])
		b = tf.get_variable(name='b', dtype=tf.float32, shape=[self.num_tags])
		
		outputs = tf.reshape(outputs, [-1, hp.num_units])
		logits = tf.matmul(outputs, w) + b
		logits = tf.reshape(logits, [-1, hp.max_len, self.num_tags])
		return logits
	
	def crf_layer(self):
		log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(self.logits, self.y, self.seq_lens)
		loss = tf.reduce_mean(-log_likelihood)
		return loss, transition
	
	def optimize(self):
		"""
		优化器
		:return:
		"""
		optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
		train_op = optimizer.minimize(self.loss, global_step=self.global_step)
		return train_op
	
	def predict(self, logits, transition, seq_lens):
		pre_seqs = []
		for score, seq_len in zip(logits, seq_lens):
			pre_seq, pre_score = tf.contrib.crf.viterbi_decode(score[:seq_len], transition)
			pre_seqs.append(pre_seq)
		return pre_seqs
