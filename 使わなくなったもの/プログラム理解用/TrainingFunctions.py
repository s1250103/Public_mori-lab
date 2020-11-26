from setting_train import *

def file_input():
    ## The function input files and makes the model.
    print("executing file_input().....")
    ## global
    #train_vid = []
    #train_label = []
    video_forTrainning = []
    label_ofVideo = []
    ## train_vid @ video_forTrainning
    ## train_label @ label_ofVideo

    #video = []
    ## 下で重複している

    #labels = []

    ## local
    ## 下で重複している。
    #count = 0
    #numFreme_one = 0

    numFrame_Whole = 0

    #for i, d in enumerate(train_vid_dirs):
    for i, each_VideoTypeDirectoty in enumerate(VideoTypeDirectoty):
        ## train_vid_dirs -> ['0.other', '1.food', '2.car', '3.cosme'] : defined in the system.
        ## loop means -> (0, '0.other'), (1, '1.food'), ..., (i, d)
        ## d は　each_VideoTypeDirectotyに変更した。


        # 各ディレクトリ内のファイル名取得
        # 0.food,  1.car  2.cosme
        type_class = os.listdir(TRAIN_DIR + each_VideoTypeDirectoty)
        ## TRAIN_DIR -> /CM_vid/ : direcroty path
        ## d -> 0.other, 1.food, ... : defined in the system
        ## files -> path/0.other, path/1.food, ...
        ## files @ type_class

        tmp = np.zeros(NUM_CLASSES)
        ## NUM_CLASSES -> len(train_vid_dirs) : 4
        ## tmp : [0, 0, 0, 0]
        tmp[i] = 1
        ## tmp : [1, 0, 0, 0]
        for f in type_class:
            ## f : path/0.other, path/1.food, ...
            #カテゴリ毎に動画を読み込む
            ## あるカテゴリの中の、１つ動画を読み込んで、扱っていく
            #cap = cv2.VideoCapture(TRAIN_DIR + each_VideoTypeDirectoty + '/' + f)
            input_video = cv2.VideoCapture(TRAIN_DIR + each_VideoTypeDirectoty + '/' + f)
            ## VideoCapture() :  動画ファイルを読み込んでいる。
            ## TRAIN_DIR + d + '/' + f -> /path/0.other/ ????
            ## cap @ input_video
            print(TRAIN_DIR + each_VideoTypeDirectoty + '/' + f)
            #video = []
            TMP_list_originated_video = []
            #video @ TMP_list_originated_video
            numFreme_one = 0
            count = 0
            while(1):
                ###
                # v        : 訓練動画全体のフレーム数
                # framenum : 動画の全フレーム数
                # count    : FPSを減らした時の動画の全フレーム数
                ###

                ## v は numFrame_Whole　に変更
                ## framenum は numFreme_one　に変更
                numFrame_Whole += 1
                numFreme_one += 1
                IO_read, img = input_video.read()
                ## cap.read() : ある１つの動画に対して、 IO_read=読み込めたかどうか(bool), img=ndarrayのタプル　を返す
                ## ret は　IO_read に変更
                if not IO_read: ## この場合は、動画が読み込めなかったことを意味する。
                    ## ループを次に進める
                    print("error")
                    break
                if numFreme_one % FPS == 0: ## もしあるCMに対して、１４フレーム、２８フレーム...のときにおいて
                    ## FPS -> 14
                    count += 1
                    ## count : あるCMを１４秒おきに切り取った回数。
                    #img = cv2.resize(img, (int(WIDTH), int(HEIGHT)))
                    img_resize = cv2.resize(img, (int(WIDTH), int(HEIGHT)))
                    ## img は　img_resize にした。なぜなら、入力のimgとは違うものであるから。
                    ## resize() : 画像を、８０ x 45 に変換している
                    ## WIDTH : 80 in the environment
                    ## HEIGHT : 45 in the environment
                    #img = img_.flatten().astype(np.float32)/255.0
                    img_flatten = img_resize.flatten().astype(np.float32)/255.0
                    ## img_resize は　img_flatten にした。なぜなら、入力のimg_resizeとは違うものであるから。
                    ## flatten().astype(np.float32)/255.0 : 画像を配列にした。それを255で正規化。

                    img_list = img_flatten.tolist()
                    ## tolist() : NumPy配列ndarrayをリスト型listに変換
                    TMP_list_originated_video.append(img_list)
                    ##　TMP_list_originated_videoに加えた。

                    #video.append(img.flatten().astype(np.float32)/255.0)
                    #print ("[%d]%s: %s : %d")%(count, d, f, i)
                    print("[{}]{}: {} : {}".format(count, each_VideoTypeDirectoty, f, i))
                    ## print 0,

                    if count == DEPTH:
                                        ## もしあるCMに対して、１４フレーム、２８フレーム...のときにおいて、
                                        ## あるCMを１４秒おきに切り取った回数が３０になったとき。
                                        ## つまり、動画から、枚画像を切り出した場合、ループから抜ける。
                        break
            if count == DEPTH:
                                ## もしあるCMに対して、１４フレーム、２８フレーム...でないときにおいて、　
                                ## かつ、　あるCMを１４秒おきに切り取った回数が３０になったとき。　
                                ## つまり、動画から、３０枚画像を切り出した場合、ループから抜ける。
                list_originated_some_videos = list_flatten(TMP_list_originated_video)
                ## list_flatten() : すでにあるCMに対して、１４フレーム、２８フレーム...おきに画像を切り取った。　そしてその画像は既にそれぞれ、配列になっている。つまり画像の枚数分（それはDEPTH=30である）の配列があるはず。それらを全て一つの配列（動画１つ分に対応している）に統合する。
                list_originated_some_videos_numpy = np.array(list_originated_some_videos)
                ## np.array() : numpyの配列に正している。
                train_vid.append(list_originated_some_videos_numpy)
                ## それをtrain_vidにくわえている。

                #train_vid.append(list_flatten(video))
            #print(train_vid)
                train_label.append(tmp)
                ## train_labelに、必ず[1,0,0,0]を入れている。




            ###
            # 0~255から0~1の範囲に変換
            # video.append(img.flatten().astype(np.float32)/255.0)
            # video[[画像1],[画像2],[画像3]...]
            # train_vid = [[動画1][動画2][動画3]...]
            # ラベルを1-of-k方式で用意する
            # train_label[foodの画像=[1,0,0],car=[0,1,0],cosme=[0,0,1]...]
            # tmp = np.zeros(NUM_CLASSES)
            # tmp[i] = 1
            # train_label.append(tmp)
            ###

            # print "[%d]%s: %s : %d"%(v, d, f, i)
        #labels.append(len(files))
        ## labels にfilsのサイズ -> 4を入れている。

    #print ("train_label:%s   train_video:%s")%(len(train_label), len(train_vid))
    #""
    print("train_label:{} train_video:{}".format(len(label_ofVideo),len(video_forTrainning)))
    #print ("DEPTH:%d       FPS:%d")%(DEPTH, FPS)
    print("DEPTH:{}      FPS:{}".format(len(label_ofVideo),len(video_forTrainning)))
    # numpy形式に変換
    video_forTrainning = np.asarray(video_forTrainning)
    ## video_forTrainningがnumpy形式に変換されている
    label_ofVideo = np.asarray(label_ofVideo)
    ## label_ofVideoがnumpy形式に変換されている
    print(video_forTrainning.shape)
    print(label_ofVideo.shape)
    #print("batch size:    %d")%(FLAGS.batch_size)
    print("batch size:     {}".format(FLAGS.batch_size))
    #print("learning rate: %g")%(FLAGS.learning_rate)
    print("leaning rate: {}".format(FLAGS.learning_rate))
    #print("activation_function: %s")%(FLAGS.act_func)
    print("activation_function: {}".format(FLAGS.act_func))

    return video_forTrainning, label_ofVideo




def training_function(train_vid, train_label):
    print("executing training.....")

    with tf.Graph().as_default():
        # 動画を入れる仮のTensor
        videos_placeholder = tf.compat.v1.placeholder("float")#

        #train_vidをvideos_placeholderに代入

        # with tf.compat.v1.Session() as sess:
        #     vdieos_placeholder = sess.run(train_vid)
        #     print(videos_placeholder)

        # ラベルを入れる仮のTensor
        labels_placeholder = tf.compat.v1.placeholder("float", shape=(None, NUM_CLASSES))

        #W = tf.Variable(tf.zeros([IMAGE_PIXELS,NUM_CLASSES]))
        #b = tf.Variable(tf.zeros([NUM_CLASSES]
        logits = inference(videos_placeholder, keep_prob, FULL_CONNECT_UNIT, NUM_CLASSES)
        # loss()を呼び出して損失を計算
        loss_value = loss(logits, labels_placeholder)
        # training()を呼び出して訓練
        train_op = training(loss_value, FLAGS.learning_rate)
        # 精度の計算
        acc = accuracy(logits, labels_placeholder)


        #### ここまでがモデルの作成、　ここから保存の準備
        ## sessは/model/年月日時間.ckptを保存している。（この中で！！）



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

        #### ここまでが保存の準備、ここからが訓練の実行

        # 訓練の実行
        log = open(LOG_NAME, 'w')
        ## log : /log/年月日時間.txt というものを作って、書き込もうとしている。

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
                save_path = saver.save(sess, MODEL_NAME, global_step=step )
                break
            elapsed_time = time.time() - start
            if elapsed_time >= FLAGS.max_time*60*60:
                save_path = saver.save(sess, MODEL_NAME, global_step=step )
                break

    print ("Elapsed Time: {0} [sec]".format(elapsed_time))

    return log, step, elapsed_time, train_accuracy


#def file_output(sess, log, train_vid, labels, step, elapsed_time, train_accuracy):
def file_output(log, train_vid, labels, step, elapsed_time, train_accuracy):
    print("executing file_output....")

    # 最終的なモデルを保存

    #save_path = saver.save(sess, MODEL_NAME)

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
    for i, d in enumerate(VideoTypeDirectoty):
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
