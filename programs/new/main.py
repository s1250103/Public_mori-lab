from TrainingFunctions import file_input
from TrainingFunctions import training
from TrainingFunctions import file_output

class main: # interface
    def make_model(self): # for training
        train_vid, train_label, video, labels = file_input()
        sess, saver, log, step, elapsed_time, train_accuracy = training(train_vid, train_label, video, labels)
        file_output(sess, saver, log, train_vid, labels, step, elapsed_time, train_accuracy)
    def tests(self): # for testing
        pass




if __name__ == '__main__':
    ps0 = main() # run main process
    ps0.make_model()
