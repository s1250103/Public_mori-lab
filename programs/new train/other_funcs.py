import tensorflow as tf

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
