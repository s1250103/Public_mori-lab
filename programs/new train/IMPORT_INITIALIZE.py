#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append("/usr/lib/python2.7/dist-packages")
import cv2
import tensorflow as tf
import tensorflow.python.platform
import math
import random
import os
import time
import datetime

import other_funcs as of

today = datetime.datetime.today()
LOG_NAME = 'log/' + today.strftime("%Y-%m-%d(%H-%M-%S)") + '.txt'
MODEL_NAME = 'model/' + today.strftime("%Y-%m-%d(%H-%M-%S)") + '.ckpt'
TEMP_DIR = 'temp/' + today.strftime("%Y-%m-%d(%H-%M-%S)")
TRAIN_DIR = '/home/moriya/Desktop/old_researched/村上/Project/nn_test/CM_vid/'  #'/home/s1240099/Desktop/Project/nn_test/CM_vid/'
COLOR_CHANNELS = 3 # RGB
WIDTH = 320/4  # 80
WIDTH2 = 80
HEIGHT = 180/4 # 45
HEIGHT2 = 45
DEPTH = 30 #入力する画像の数
IMAGE_PIXELS = DEPTH*HEIGHT*WIDTH*COLOR_CHANNELS
MAX_ACCURACY = 0.9
FULL_CONNECT_UNIT = 1024
FPS = 14 #FPSの数ごとに一枚画像を取り出す

flags = tf.app.flags
FLAGS = flags.FLAGS

# 画像のあるディレクトリ
# train_img_dirs = ['0.other', '1.food', '2.car', '3.cosme', '4.drug', '5.movie', '6.game', '7.phone', '8.clean']
train_vid_dirs = ['0.other', '1.food', '2.car', '3.cosme']
NUM_CLASSES = len(train_vid_dirs)

# flags.DEFINE_string('train', 'train.txt', 'File name of train data')
#flags.DEFINE_string('test', 'test2.txt', 'File name of test data')
flags.DEFINE_string('train_dir', TEMP_DIR, 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
# flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 24, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 1, 'Batch size'
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-5, 'Initial learning rate.')
flags.DEFINE_integer('channel1', 8 , 'Number of conv1 & conv2 channel')
flags.DEFINE_integer('channel2', 16, 'Number of conv3 & conv4 channel')
flags.DEFINE_integer('channel3', 32, 'Number of conv5 & conv6 channel')
flags.DEFINE_integer('channel4', 64, 'Number of conv7 & conv8 channel')
flags.DEFINE_integer('convs', 3, 'size of conv height & width')
flags.DEFINE_integer('convd', 3, 'size of conv depth')
flags.DEFINE_string('act_func', 'relu', 'activation function')
flags.DEFINE_integer('max_time', 3, 'training time')
