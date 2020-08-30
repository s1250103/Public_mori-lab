#!/usr/bin/env python
# -*- coding: utf-8 -*-
import global


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
