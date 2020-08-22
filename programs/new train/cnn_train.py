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


today = datetime.datetime.today()
LOG_NAME = 'log/' + today.strftime("%Y-%m-%d(%H-%M-%S)") + '.txt'
MODEL_NAME = 'model/' + today.strftime("%Y-%m-%d(%H-%M-%S)") + '.ckpt'
TEMP_DIR = 'temp/' + today.strftime("%Y-%m-%d(%H-%M-%S)")
TRAIN_DIR = '~/Desktop/old_researched/村上/Project/nn_test/CM_vid/'  #'/home/s1240099/Desktop/Project/nn_test/CM_vid/'
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

def loss(logits, labels):
    """ lossを計算する関数

    引数:
      logits: ロジットのtensor, float - [batch_size, NUM_CLASSES]
      labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]

    返り値:
      cross_entropy: 交差エントロピーのtensor, float

    """

    # 交差エントロピーの計算
    cross_entropy = -tf.reduce_sum(labels*tf.math.log(tf.clip_by_value(logits,1e-12,1.0)),reduction_indices=[1])

    # TensorBoardで表示するよう指定
    tf.compat.v1.summary.scalar("cross_entropy", cross_entropy)
    print(cross_entropy)
    return cross_entropy


def training(loss, learning_rate):
    """ 訓練のOpを定義する関数

    引数:
      loss: 損失のtensor, loss()の結果
      learning_rate: 学習係数

    返り値:
      train_step: 訓練のOp

    """

    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step


def accuracy(logits, labels):
    """ 正解率(accuracy)を計算する関数

    引数:
      logits: inference()の結果
      labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]

    返り値:
      accuracy: 正解率(float)

    """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.compat.v1.summary.scalar("accuracy",accuracy)
    return accuracy

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

    # データを入れる配列
    train_vid = []
    train_label = []
    train_tmp = []
    video = []
    count = 0
    framenum = 0

    labels = []

    # png = '.png'
    v = 0
    for i, d in enumerate(train_vid_dirs):
        # 各ディレクトリ内のファイル名取得
        # 0.food,  1.car  2.cosme
        files = os.listdir(TRAIN_DIR + d)
        tmp = np.zeros(NUM_CLASSES)
        tmp[i] = 1
        for f in files:
            #カテゴリ毎に動画を読み込む
            cap = cv2.VideoCapture(TRAIN_DIR + d + '/' + f)
            print(TRAIN_DIR + d + '/' + f)
            video = []
            framenum = 0
            count = 0
            while(1):
                """
                v        : 訓練動画全体のフレーム数
                framenum : 動画の全フレーム数
                count    : FPSを減らした時の動画の全フレーム数
                """
                v += 1
                framenum += 1
                ret, img = cap.read()
                # print(ret)
                if not ret:
                    print("error")
                    # sys.exit()
                    break
                if framenum % FPS == 0:
                    count += 1
                    img = cv2.resize(img, (int(WIDTH), int(HEIGHT)))
                    # cv2.imwrite("test.png", img)
                    img = img.flatten().astype(np.float32)/255.0
                    img = img.tolist()
                    video.append(img)
                    #video.append(img.flatten().astype(np.float32)/255.0)
                    #print ("[%d]%s: %s : %d")%(count, d, f, i)
                    print("[{}]{}: {} : {}".format(count, d, f, i))

                    if count == DEPTH:
                        break
            if count == DEPTH:
                video = list_flatten(video)
                video = np.array(video)
                train_vid.append(video)
                #train_vid.append(list_flatten(video))
	        #print(train_vid)
                train_label.append(tmp)




            """
            0~255から0~1の範囲に変換
            video.append(img.flatten().astype(np.float32)/255.0)
            video[[画像1],[画像2],[画像3]...]
            train_vid = [[動画1][動画2][動画3]...]
            ラベルを1-of-k方式で用意する
            train_label[foodの画像=[1,0,0],car=[0,1,0],cosme=[0,0,1]...]
            tmp = np.zeros(NUM_CLASSES)
            tmp[i] = 1
            train_label.append(tmp)
            """

            # print "[%d]%s: %s : %d"%(v, d, f, i)
        labels.append(len(files))

    #print ("train_label:%s   train_video:%s")%(len(train_label), len(train_vid))
    #""
    print("train_label:{} train_video:{}".format(len(train_label),len(train_vid)))
    #print ("DEPTH:%d       FPS:%d")%(DEPTH, FPS)
    print("DEPTH:{}      FPS:{}".format(len(train_label),len(train_vid)))
    # numpy形式に変換
    train_vid = np.asarray(train_vid)
    train_label = np.asarray(train_label)
    print(train_vid.shape)
    print(train_label.shape)
    #print("batch size:    %d")%(FLAGS.batch_size)
    print("batch size:     {}".format(FLAGS.batch_size))
    #print("learning rate: %g")%(FLAGS.learning_rate)
    print("leaning rate: {}".format(FLAGS.learning_rate))
    #print("activation_function: %s")%(FLAGS.act_func)
    print("activation_function: {}".format(FLAGS.act_func))

    with tf.Graph().as_default():
        # 動画を入れる仮のTensor
        videos_placeholder = tf.compat.v1.placeholder("float", shape=(None, IMAGE_PIXELS))

        #train_vidをvideos_placeholderに代入

        # with tf.compat.v1.Session() as sess:
        #     vdieos_placeholder = sess.run(train_vid)
        #     print(videos_placeholder)

        # ラベルを入れる仮のTensor
        labels_placeholder = tf.compat.v1.placeholder("float", shape=(None, NUM_CLASSES))

        W = tf.Variable(tf.zeros([IMAGE_PIXELS,NUM_CLASSES]))
        b = tf.Variable(tf.zeros([NUM_CLASSES]))

        keep_prob = tf.compat.v1.placeholder("float")
        # inference()を呼び出してモデルを作る
        logits = inference(videos_placeholder, keep_prob)
        # loss()を呼び出して損失を計算
        loss_value = loss(logits, labels_placeholder)
        # training()を呼び出して訓練
        train_op = training(loss_value, FLAGS.learning_rate)
        # 精度の計算
        acc = accuracy(logits, labels_placeholder)

        # 保存の準備
        #saver = tf.train.Saver()
        saver = tf.compat.v1.train.Saver()
        # Sessionの作成
        #sess = tf.Session()
        sess = tf.compat.v1.Session()
        # 変数の初期化
        #sess.run(tf.global_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())
        # TensorBoardで表示する値の設定
        os.mkdir(FLAGS.train_dir)
        #summary_op = tf.summary.merge_all()
        summary_op = tf.compat.v1.summary.merge_all()
        #summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        summary_writer = tf.compat.v1.summary.FileWriter(FLAGS.train_dir, sess.graph)

        # 訓練の実行
        log = open(LOG_NAME, 'w')
        log.write('Start Time: ')
        today = datetime.datetime.today()
        log.write(today.strftime("%Y/%m/%d %H:%M:%S"))
        log.write("\n")
        start = time.time()
        for step in range(FLAGS.max_steps):
            for i in range(len(train_vid)//FLAGS.batch_size):# batch_size分の動画に対して訓練の実行(ここでは14回)
                batch = FLAGS.batch_size*i#ここは動画のfpsの都合上、気にしなくて良いもの
                # feed_dictでplaceholderに入れるデータを指定する
                # placeholderは変数のようなもの
                sess.run(train_op, feed_dict={
                  videos_placeholder: train_vid[batch:batch+FLAGS.batch_size],
                  labels_placeholder: train_label[batch:batch+FLAGS.batch_size], keep_prob:0.5})
            if batch+FLAGS.batch_size != len(train_vid):#ここも動画のfpsの都合、考えなくて良いもの
                 sess.run(train_op, feed_dict={
                  videos_placeholder: train_vid[batch+FLAGS.batch_size+1:len(train_vid)],
                  labels_placeholder: train_label[batch+FLAGS.batch_size+1:len(train_vid)], keep_prob:0.5})


            # 1 step終わるたびに精度を計算する
            #動画14個を読み込むごとにstepを1回増
            train_accuracy = sess.run(acc, feed_dict={
                videos_placeholder: train_vid,
                labels_placeholder: train_label,
                keep_prob:1.0})
            #print ("step %d, training accuracy %g")%(step+1, train_accuracy)
            print("step {}, training accuracy {}".format(step+1, train_accuracy))

            # 1 step終わるたびにTensorBoardに表示する値を追加する
            # summary_str = sess.run(summary_op, feed_dict={
            #     videos_placeholder: train_vid,
            #     labels_placeholder: train_label,
            #     keep_prob:1.0})
            # summary_writer.add_summary(summary_str, step)

            if train_accuracy >= MAX_ACCURACY:
                break
            elapsed_time = time.time() - start
            if elapsed_time >= FLAGS.max_time*60*60:
                break

    print ("Elapsed Time: {0} [sec]".format(elapsed_time))
    # 最終的なモデルを保存
    save_path = saver.save(sess, MODEL_NAME)

    print (FLAGS.train_dir)

    log.write('Finish Time: ')
    today  = datetime.datetime.today()
    log.write(today.strftime("%Y/%m/%d %H:%M:%S"))
    log.write("\n")
    log.write("NUM_CLASSES: %d\n"%(NUM_CLASSES))
    log.write("activation_function: %s\n"%(FLAGS.act_func))
    log.write("COLOR_CHANNELS: %d\n"%(COLOR_CHANNELS))
    log.write("WIDTH: %d\n"%(WIDTH))
    log.write("HEIGHT: %d\n"%(HEIGHT))
    log.write("IMAGE_PIXELS: %d\n"%(IMAGE_PIXELS))
    log.write("MAX_ACCURACY: %f\n"%(MAX_ACCURACY))
    log.write("Train Video Total: %d\n"%(len(train_vid)))
    log.write('Train Video Directories: ')
    for i, d in enumerate(train_vid_dirs):
        log.write("%s(%d), "%(d, labels[i]))
    log.write("\n")
    log.write("Max Steps: %d\n"%(FLAGS.max_steps))
    log.write("chanel 1: %d\n"%(FLAGS.channel1))
    log.write("chanel 2: %d\n"%(FLAGS.channel2))
    log.write("chanel 3: %d\n"%(FLAGS.channel3))
    log.write("Batch Size: %d\n"%(FLAGS.batch_size))
    log.write("Learning Rate: %f\n\n"%(FLAGS.learning_rate))
    log.write("Final Step: %d\n"%(step+1))
    log.write("Training Accuracy %g\n"%(train_accuracy))
    log.write("MODEL_NAME: %s\n"%(MODEL_NAME))
    log.write("Elapsed Time: {0} [sec]\n\n".format(elapsed_time))

    log.close()
