#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
import cv2
import tensorflow as tf
import tensorflow.python.platform
import math
import random
import os
import time
import datetime

from tensorflow.contrib import rnn


args = sys.argv
today = datetime.datetime.today()


MODEL_NAME = 'model/2019-12-18(01-40-41).ckpt'

TEST_DIR = '../Project/nn_test/test_vid/'
TEMP_DIR = 'run/' + today.strftime("%Y-%m-%d(%H-%M-%S)") + '/'
TEXT_NAME = 'log.txt'
COLOR_CHANNELS = 3 # RGB
WIDTH = 320/4
HEIGHT = 180/4
WIDTH2 = 80
HEIGHT2 = 45
FPS = 14
FULL_CONNECT_UNIT = 1024
DEPTH = 30 #入力する画像の数
IMAGE_PIXELS = DEPTH*HEIGHT*WIDTH*COLOR_CHANNELS

flags = tf.app.flags
FLAGS = flags.FLAGS

test_dirs = ['0.other', '1.food', '2.car', '3.cosme']

NUM_CLASSES = len(test_dirs)

# flags.DEFINE_string('train', 'train.txt', 'File name of train data')
# flags.DEFINE_string('test', 'test2.txt', 'File name of test data')
flags.DEFINE_string('train_dir', 'temp', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
#flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
#flags.DEFINE_integer('hidden2', 24, 'Number of units in hidden layer 2.')
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

def inference(videos_placeholder, keep_prob):
    """ 予測モデルを作成する関数

    引数:
      videos_placeholder: 画像のplaceholder
      keep_prob: dropout率のplaceholder

    返り値:
      y_conv: 各クラスの確率(のようなもの)

     with tf.name_scope("xxx") as scope:
         これでTensorBoard上に一塊のノードとし表示される

    conv3d(input, filer, strides, padding)
      input(batch, depth, height, width, channels)
      filter(depth, height, width, in_channels, out_channels)
      strides[1, x, x, x, 1]
      padding=SAMEゼロパディング
    """


    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
        inital = tf.random.truncated_normal(shape,stddev=0.1)
        return tf.Variable(inital)

    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
        inital = tf.constant(0.1, shape=shape)
        return tf.Variable(inital)

    # 畳み込み層 ストライド1
    def conv3d(x, W):
        return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding="SAME")

    # プーリング層 ストライド2
    def max_pool_2x2x2(x):
        return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="SAME")

    # 入力層をDEPTH*HEIGHT*WIDTH*COLOR_CHANNELに変形
    #ここx_videoで５次元配列にしていますね...。
    print(videos_placeholder)
    x_video = tf.reshape(videos_placeholder, [-1, DEPTH, HEIGHT2, WIDTH2, COLOR_CHANNELS])
    #print("hoge",x_video.shape)

    # 畳み込み層1の作成
    with tf.name_scope("conv1") as scope:
        W_conv1 = weight_variable([FLAGS.convd,FLAGS.convs,FLAGS.convs,COLOR_CHANNELS,FLAGS.channel1])
        b_conv1 = bias_variable([FLAGS.channel1])
        h_conv1 = activation_function(conv3d(x_video, W_conv1) + b_conv1)
        mb, d, h, w, c = h_conv1.get_shape().as_list()
        print("conv1      d:{} h:{} w:{} c:{}".format(d, h, w, c))

    # プーリング層1の作成
    with tf.name_scope("pool1") as scope:
        h_pool1 = max_pool_2x2x2(h_conv1)
        mb, d, h, w, c = h_pool1.get_shape().as_list()
        print("pool1      d:{} h:{} w:{} c:{}".format(d, h, w, c))

    # 畳み込み層2の作成
    with tf.name_scope("conv2") as scope:
        W_conv2 = weight_variable([FLAGS.convd,FLAGS.convs,FLAGS.convs,FLAGS.channel1,FLAGS.channel2])
        b_conv2 = bias_variable([FLAGS.channel2])
        h_conv2 = activation_function(conv3d(h_pool1, W_conv2) + b_conv2)
        mb, d, h, w, c = h_conv2.get_shape().as_list()
        print("conv2      d:{} h:{} w:{} c:{}".format(d, h,  w ,  c ))


    # プーリング層2の作成
    with tf.name_scope("pool2") as scope:
        h_pool2 = max_pool_2x2x2(h_conv2)
        mb, d, h, w, c = h_pool2.get_shape().as_list()
        print("pool2      d:{} h:{} w:{} c:{}".format( d ,  h ,  w ,  c ))

    # 畳み込み層3の作成
    with tf.name_scope("conv3") as scope:
        W_conv3 = weight_variable([FLAGS.convd,FLAGS.convs,FLAGS.convs,FLAGS.channel2,FLAGS.channel3])
        b_conv3 = bias_variable([FLAGS.channel3])
        h_conv3 = activation_function(conv3d(h_pool2, W_conv3) + b_conv3)
        mb, d, h, w, c = h_conv3.get_shape().as_list()
        print("conv3     d:{} h:{} w:{} c:{}".format( d , h,  w ,  c ))

    # プーリング層3の作成
    with tf.name_scope("pool3") as scope:
        h_pool3 = max_pool_2x2x2(h_conv3)
        mb, d, h, w, c = h_pool3.get_shape().as_list()
        print("pool3      d:{} h:{} w:{} c:{}".format( d ,  h ,  w ,  c ))

    # 畳み込み層4の作成
    with tf.name_scope("conv4") as scope:
        W_conv4 = weight_variable([FLAGS.convd,FLAGS.convs,FLAGS.convs,FLAGS.channel3,FLAGS.channel4])
        b_conv4 = bias_variable([FLAGS.channel4])
        h_conv4 = tf.nn.relu(conv3d(h_pool3, W_conv4) + b_conv4)
        mb, d, h, w, c = h_conv4.get_shape().as_list()
        print("conv4     d:{} h:{} w:{} c:{}".format(d, h, w, c))


    # プーリング層4の作成
    with tf.name_scope("pool4") as scope:
        h_pool4 = max_pool_2x2x2(h_conv4)
        mb, d, h, w, c = h_pool4.get_shape().as_list()
        print("pool4      d:{} h:{} w:{} c:{}".format(d, h, w, c))


        """
     #畳み込み層5aの作成
    with tf.name_scope("conv5a") as scope:
        W_conv5a = weight_variable([FLAGS.convd,FLAGS.convs,FLAGS.convs,FLAGS.channel4,FLAGS.channel4])
        b_conv5a = bias_variable([FLAGS.channel4])
        h_conv5a = tf.nn.relu(conv3d(h_pool4, W_conv5a) + b_conv5a)
        mb, d, h, w, c = h_conv5a.get_shape().as_list()
        print "conv5a     d:%d h:%d w:%d c:%d"%(d, h, w, c)

     #畳み込み層5bの作成
    with tf.name_scope("conv5b") as scope:
        W_conv5b = weight_variable([FLAGS.convd,FLAGS.convs,FLAGS.convs,FLAGS.channel4,FLAGS.channel4])
        b_conv5b = bias_variable([FLAGS.channel4])
        h_conv5b = tf.nn.relu(conv3d(h_conv5a, W_conv5b) + b_conv5b)
        mb, d, h, w, c = h_conv5b.get_shape().as_list()
        print "conv5b     d:%d h:%d w:%d c:%d"%(d, h, w, c)

     #プーリング層5の作成
    with tf.name_scope("pool5") as scope:
        h_pool5 = max_pool_2x2x2(h_conv5b)
        mb, d, h, w, c = h_pool5.get_shape().as_list()
        print "pool5      d:%d h:%d w:%d c:%d"%(d, h, w, c)
        """
    # 結合層1の作成
    with tf.name_scope("fc1") as scope:
        mb, d, h, w, c = h_pool4.get_shape().as_list()
        #"pool5      d:{} h:{} w:{} c:{}".format("d", "h", "w", "c")
        W_fc1 = weight_variable([d*h*w*c, FULL_CONNECT_UNIT])#前者から後者のヘッジをつなげた。
        b_fc1 = bias_variable([FULL_CONNECT_UNIT])
        h_pool_flat = tf.reshape(h_pool4, [-1, d*h*w*c])
        h_fc1 = activation_function(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
        # dropout1の設定
        h_fc_1_drop = tf.nn.dropout(h_fc1, rate=1-keep_prob)


    # 結合層2の作成
    with tf.name_scope("fc2") as scope:
        W_fc2 = weight_variable([FULL_CONNECT_UNIT, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    # ソフトマックス関数による正規化
    with tf.name_scope("softmax") as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc_1_drop, W_fc2) + b_fc2)

    # 各ラベルの確率のようなものを返す
    return y_conv

def activation_function(x):
    if FLAGS.act_func == 'relu':
        y = tf.nn.relu(x)
    elif FLAGS.act_func == 'selu':
        y = tf.nn.selu(x)
    elif FLAGS.act_func == 'elu':
        y = tf.nn.elu(x)
    elif FLAGS.act_func == 'tanh':
        y = tf.nn.tanh(x)
    elif FLAGS.act_func == 'sigmoid':
        y = tf.nn.sigmoid(x)
    elif FLAGS.act_func == 'crelu':
        y = tf.nn.crelu(x)
    elif FLAGS.act_func == 'leaky_relu':
        y = tf.nn.leaky_relu(x)
    elif FLAGS.act_func == 'relu6':
        y = tf.nn.relu6(x)

    return y


def list_flatten(x):
    """
    x = [[1,2],[3],[4,5,6]] → [1,2,3,4,5,6]

    """
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(list_flatten(el))
        else:
            result.append(el)
    return result



if __name__ == '__main__':


    test_video = []
    test_label = []
    #temp_video = []
    test_name = []
    video = []
    framenum = 0
    count = 0

    for i, d in enumerate(test_dirs):
        files = os.listdir(TEST_DIR + d)
        tmp = np.zeros(NUM_CLASSES)
        tmp[i] = 1

        for f in files:
            cap = cv2.VideoCapture(TEST_DIR + d + '/' + f)
            video = []
            framenum = 0
            count = 0
            test_name.append(f)

            while(1):
                framenum += 1
                ret, img = cap.read()
                if not ret:
    	            break
                if framenum%FPS == 0:
                    count += 1
                    img = cv2.resize(img, (int(WIDTH), int(HEIGHT)))
                    #temp_video.append(img)
                    video.append(img.flatten().astype(np.float32)/255.0)
                    #print "[%d]%s: %d"%(count, f, framenum)
                    print("[{}]{}: {}".format(count,f,framenum))
                    if count == DEPTH:
                        break
            if count == DEPTH:
                test_video.append(list_flatten(video))
                test_label.append(tmp)

    test_video = np.asarray(test_video)
    print(test_label[i])
    print("activation_function: {}".format(FLAGS.act_func))

    with tf.Graph().as_default():
        # 画像を入れる仮のTensor
        videos_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        # ラベルを入れる仮のTensor
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))

        keep_prob = tf.placeholder("float")
        # inference()を呼び出してモデルを作る
        logits = inference(videos_placeholder, keep_prob)

        # 保存の準備
        saver = tf.train.Saver()
        # Sessionの作成
        sess = tf.InteractiveSession()
        # 変数の初期化
        sess.run(tf.global_variables_initializer())

    saver.restore(sess, MODEL_NAME)

    os.mkdir(TEMP_DIR)
    for i in range(NUM_CLASSES):
        os.mkdir(TEMP_DIR + "/" + str(i))
    txt = open(TEMP_DIR + TEXT_NAME, 'w')
    txt.write("TEST_DIR: " + TEST_DIR + "\n")
    txt.write("MODEL_NAME: " + MODEL_NAME + "\n")
    txt.write("NUM_CLASSES: %d\n"%(NUM_CLASSES))
    txt.write("COLOR_CHANNELS: %d\n"%(COLOR_CHANNELS))
    txt.write("WIDTH: %d\n"%(WIDTH))
    txt.write("HEIGHT: %d\n"%(HEIGHT))
    txt.write("IMAGE_PIXELS: %d\n\n"%(IMAGE_PIXELS))
    txt.write("activation_function: %s\n"%(FLAGS.act_func))


    cate = []
    label = []
    count = 0
    ok_sum = 0
    # cate = [0, 0, 0, 0] 4category
    for i in range(NUM_CLASSES):
        cate.append(0)
    # 動画の数分ループ
    for i in range(len(test_video)):
        print(test_name[i])
        accr = logits.eval(session=sess, feed_dict={
                videos_placeholder: [test_video[i]], keep_prob:1.0})[0]
        # pred = nnの出力 0,1,2,3
        pred = np.argmax(accr)
        tmp = np.zeros(NUM_CLASSES)
        tmp[pred] = 1
        print(tmp)
        txt.write("[%s](%d): %d\n"%(test_name[i], i, pred))
        ssss = 0
        for c in range(NUM_CLASSES):
            txt.write("%f "%(accr[c]))
            print(accr[c])
            ssss = ssss + accr[c] # ssss is the sum of each accr
        txt.write("  all: %f"%ssss) # edit by moriya
        txt.write("\n\n")
        #print "testing (%d/%d)"%(i+1, len(test_video))
        print("testing ({}/{})".format(i+1, len(test_video)))
        #category分ループ
        for j in range(len(cate)):
            if pred == j:
                cate[j] = cate[j] + 1
                if j == 0:
                    print("output: other")
                elif j == 1:
                    print("output: food")
                elif j == 2:
                    print("output: car")
                elif j == 3:
                    print("output: cosme")
                #cv2.imwrite(TEMP_DIR + "/" + str(j) + "/" + str(i) + ".png", temp_video[i])
        if((test_label[i] == tmp).all()):
            print("OK\n")
            ok_sum += 1
        else:
            print("NG\n")

#@ the number, just dividing OK (recognited) by all (including not OK).
    txt.write("accuracy rate: %f %% [%d/%d]\n"%(float(ok_sum) / len(test_video) * 100, ok_sum, len(test_video)))
    #print("accuracy rate: %f %% [%d/%d]"%(float(ok_sum) / len(test_video) * 100, ok_sum, len(test_video)))

    print("accuracy rate: {} %% [{}/{}]".format(float(ok_sum) / len(test_video) * 100, ok_sum, len(test_video)))


    txt.close()
