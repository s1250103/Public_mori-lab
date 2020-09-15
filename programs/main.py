 # modules for training
from TrainingFunctions import file_input
from TrainingFunctions import training_function
from TrainingFunctions import file_output
# modules for testing
from TestingFunctions import initialize
from TestingFunctions import execution


class main: # interface
    def make_model(self): # for training
        #train_vid, train_label = file_input()
        video_forTrainning, label_ofVideo = file_input()
        ## video_forTrainning : 入力として動画を、訓練機能に渡される
        ## label_ofVideo : 動画に対するラベルは、訓練機能に渡される
        #log, step, elapsed_time, train_accuracy = training_function(train_vid, train_label)
        log, step, elapsed_time, train_accuracy = training_function(video_forTrainning, label_ofVideo)
        -> log = training_function(video_forTrainning, label_ofVideo)
        ## log : ファイルに書き込むための変数は、出力機能に渡される。
        ## step : FPS=14と同じもの。　FPSと変えても問題ない。本質的には渡す必要はない。
        ## elapsed_time : 訓練の経過時間は　log に書き込むだけのために渡される。本質的には渡す必要はない。
        ## train_accuracy  : 訓練の正確さ（設定した閾値を超えたもの）は、logに書き込むだけのために渡される。本質的には渡す必要はない。　
        ## log のみでいい。
        file_output(log, video_forTrainning, label_ofVideo, step, elapsed_time, train_accuracy)
        -> file_outpuf(log, video_forTrainning, SIZE_InputVideo)
        ## video_forTrainning : 入力した動画の数量のみ必要なので、動画そのものを渡す必要はない。
        ## label_ofVideo : ラベルは表示するために渡される。
        ## step : FPS=14と同じもの。　FPSと変えても問題ない。本質的には渡す必要はない。
        ## elapsed_time : 訓練の経過時間は　log に書き込むだけのために渡される。本質的には渡す必要はない。
        ## train_accuracy  : 訓練の正確さ（設定した閾値を超えたもの）は、logに書き込むだけのために渡される。本質的には渡す必要はない。　
        ## label_ofVideo　と　入力した動画の数量の情報のみでいい。

    def test(self): # for testing
        initialize()
        execution()




if __name__ == '__main__':
    ps0 = main() # run main process

    ps0.make_model()
