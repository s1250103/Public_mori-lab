from setting_test import *



def initialize():
    print("executing initialize().....")

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

def execution():
    print("executing execution().....")

    with tf.Graph().as_default():
        # 画像を入れる仮のTensor
        videos_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        # ラベルを入れる仮のTensor
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))

        keep_prob = tf.placeholder("float")
        # inference()を呼び出してモデルを作る
        logits = inference(videos_placeholder, keep_prob, FULL_CONNECT_UNIT, FULL_CONNECT_UNIT)

        # 保存の準備
        saver = tf.compat.v1.train.Saver()
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
