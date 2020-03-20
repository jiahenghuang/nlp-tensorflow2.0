# -*- coding: utf-8 -*-
# Brief: Use CGED corpus
import os

# CGED chinese corpus
raw_train_paths = [
    '../data/cn/CGED/CGED16_HSK_TrainingSet.xml',
    '../data/cn/CGED/sample_HSK_TrainingSet.xml',
]
output_dir = 'output'
# Training data path.
train_path = os.path.join(output_dir, 'train.txt')
# Validation data path.
test_path = os.path.join(output_dir, 'test.txt')

# seq2seq_attn_train config
save_vocab_path = os.path.join(output_dir, 'vocab.txt')
attn_model_path = os.path.join(output_dir, 'attn_model.weight')

# config
batch_size = 64
epochs = 1 #40
rnn_hidden_dim = 128
maxlen = 400
dropout = 0.0
use_gpu = False

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
