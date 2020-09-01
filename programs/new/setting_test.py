from setting_general import *

MODEL_NAME = '/home/moriya/Desktop/old_researched/村上/murakami/model/2019-12-18(01-40-41).ckpt'

TEST_DIR = '/home/moriya/Desktop/old_researched/村上/Project/nn_test/test_vid/'
TEMP_DIR = '/home/moriya/Desktop/old_researched/村上/murakami/run/' + today.strftime("%Y-%m-%d(%H-%M-%S)") + '/'
TEXT_NAME = 'log.txt'

test_dirs = ['0.other', '1.food', '2.car', '3.cosme']
NUM_CLASSES = len(test_dirs)

# flags.DEFINE_string('train', 'train.txt', 'File name of train data')
# flags.DEFINE_string('test', 'test2.txt', 'File name of test data')flags.DEFINE_string('train_dir', 'temp', 'Directory to put the training data.')
