 # modules for training
from TrainingFunctions import file_input
from TrainingFunctions import training_function
from TrainingFunctions import file_output
# modules for testing
from TestingFunctions import initialize
from TestingFunctions import execution


class main: # interface
    def make_model(self): # for training
        train_vid, train_label, video, labels = file_input()
        sess, saver, log, step, elapsed_time, train_accuracy = training_function(train_vid, train_label, video, labels)
        file_output(sess, saver, log, train_vid, labels, step, elapsed_time, train_accuracy)

    def test(self): # for testing
        initialize()
        execution()




if __name__ == '__main__':
    ps0 = main() # run main process

    ps0.make_model()
