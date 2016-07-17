import minpy.numpy as np
import utils
import optimizer
import flags

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
        if not N % self.batch_size == 0:
            print 'Illegal Batch Size'
            return
        num_batch = N // self.batch_size
        optimize = getattr(__import__('optimizer'), self.optimizer)

        accuracy_record = [0.0]
        loss_record = []
        param = []
        validation_flag = x_val.shape[0] > 0
        flags.RECORD_FLAG = False

        for epoch in range(self.epochs):
            flags.EPOCH = epoch
            flags.MODE = 'Train'
            for batch in range(num_batch):
                data = x_train[batch * self.batch_size:(batch+1) * self.batch_size]
                label = y_train[batch * self.batch_size:(batch+1) * self.batch_size]
                gradient, loss = self.model.loss(data, label)
                for p in range(len(self.model.param)):
                    self.model.param[p] = optimize(self.model.param[p], gradient[p], **self.update_setting)
                loss_record.append(loss.asnumpy())
                if batch % self.batch_size == 0:
                    print 'epoch %d batch %d loss: %f' % (epoch, batch, loss.val)
            flags.MODE = 'Test'
            if validation_flag:
                flags.RECORD_FLAG = True 
                validation_accuracy = utils.get_accuracy(np.argmax(self.model.loss(x_val),axis=1), y_val)
                print 'validation accuracy: %f' % (validation_accuracy)
                if validation_accuracy > max(accuracy_record):
                    param = [np.copy(p) for p in self.model.param]
                accuracy_record.append(validation_accuracy)
                self.model.param = [np.copy(p) for p in param]
                flags.RECORD_FLAG = False
                test_accuracy = utils.get_accuracy(np.argmax(self.model.loss(x_test), axis=1), y_test)
            else:
                test_accuracy = utils.get_accuracy(np.argmax(self.model.loss(x_test), axis=1), y_test)
                if test_accuracy > max(accuracy_record):
                    param = [np.copy(p) for p in self.model.param]
                accuracy_record.append(test_accuracy)
                self.model.param = [np.copy(p) for p in param]
            print 'test accuracy: %f' % test_accuracy
            print 'optimal accuracy: %f' % max(accuracy_record)

            if (epoch + 1) % self.decay_interval == 0:
                self.update_setting['learning_rate'] *= self.decay_rate
                print 'learning rate decayed to %f' % self.update_setting['learning_rate']
        utils.record_loss(loss_record)

        return accuracy_record[1:]
