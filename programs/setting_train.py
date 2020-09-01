from setting_general import *
sys.path.append("/usr/lib/python2.7/dist-packages")

MAX_ACCURACY = 0.9

LOG_NAME = '/home/moriya/Desktop/old_researched/村上/murakami/log/' + today.strftime("%Y-%m-%d(%H-%M-%S)") + '.txt'
MODEL_NAME = '/home/moriya/Desktop/old_researched/村上/murakami/model/' + today.strftime("%Y-%m-%d(%H-%M-%S)") + '.ckpt'
TEMP_DIR = '/home/moriya/Desktop/old_researched/村上/murakami/temp/' + today.strftime("%Y-%m-%d(%H-%M-%S)")
TRAIN_DIR = '/home/moriya/Desktop/old_researched/村上/Project/nn_test/CM_vid/'  #'/home/s1240099/Desktop/Project/nn_test/CM_vid/'

# 画像のあるディレクトリ
# train_img_dirs = ['0.other', '1.food', '2.car', '3.cosme', '4.drug', '5.movie', '6.game', '7.phone', '8.clean']
train_vid_dirs = ['0.other', '1.food', '2.car', '3.cosme']
NUM_CLASSES = len(train_vid_dirs)
flags.DEFINE_string('train_dir', TEMP_DIR, 'Directory to put the training data.')
# flags.DEFINE_string('train', 'train.txt', 'File name of train data')
#flags.DEFINE_string('test', 'test2.txt', 'File name of test data')
#flags.DEFINE_string('train_dir', TEMP_DIR, 'Directory to put the training data.')
