import minpy.numpy as np
import utils
import optimizer

class Solver:
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.data = data
        self.epochs = kwargs.pop('epoch', 30)
        self.batch_size = kwargs.pop('batch_size', 50)
        self.decay_rate = kwargs.pop('decay_rate', 0.9)
        self.decay_interval = kwargs.pop('decay_interval', 5)
        self.optimizer = kwargs.pop('optimizer', 'sgd')
        self.update_setting = kwargs.pop('update_setting', None)

    def train(self):
        x_train = self.data[0]
        y_train = self.data[1]
        x_val = self.data[2]
        y_val = self.data[3]
        x_test = self.data[4]
        y_test = self.data[5]
        N = x_train.shape[0]
        print x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape
        if not N % self.batch_size == 0:
            print 'Illegal Batch Size'
            return
        num_batch = N // self.batch_size
        optimize = getattr(__import__('optimizer'), self.optimizer)
        accuracy_record = [0.0]
        param = []

        for epoch in range(self.epochs):
            for batch in range(num_batch):
                data = x_train[batch * self.batch_size:(batch+1) * self.batch_size]
                label = y_train[batch * self.batch_size:(batch+1) * self.batch_size]
                gradient, loss = self.model.loss(data, label)
                optimize(self.model.param, gradient, **self.update_setting)
                if batch % 50 == 0:
                    print 'epoch %d batch %d loss: %f' % (epoch, batch, loss.val)
            validation_accuracy = utils.accuracy(np.argmax(self.model.loss(x_val),axis=1), y_val)
            print 'epoch %d validation accuracy: %f' % (epoch, validation_accuracy)
            if validation_accuracy > max(accuracy_record):
                param = [np.copy(p) for p in self.model.param]
            accuracy_record.append(validation_accuracy)

            if (epoch + 1) % self.decay_interval == 0:
                self.update_setting['learning_rate'] *= self.decay_rate
                print 'learning rate decayed to %f' % self.update_setting['learning_rate']

            print 'optimal accuracy: %f' % max(accuracy_record)
            self.model.param = [np.copy(p) for p in param]
            test_accuracy = utils.accuracy(np.argmax(self.model.loss(x_test), axis=1), y_test)
            print 'test accuracy: %f' % test_accuracy

        return accuracy_record
