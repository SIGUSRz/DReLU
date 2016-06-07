import cPickle as pickle
import struct 
import numpy as np
import os
def _load_cifar_data(data_dir):
    if not os.path.exists('Datasets'):
        os.system('mkdir Datasets/')
    if not os.path.exists(data_dir):
        os.system('mkdir ' + data_dir)
        os.chdir(data_dir)
        os.system('wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
        os.system('tar -xzvf cifar-10-python.tar.gz')
        os.system('rm cifar-10-python.tar.gz')
        os.chdir('..')
        os.chdir('..') 

def load_CIFAR10_batch(filename):
    #load single batch in Cifar-10
    with open(filename,'rb') as file:
        datadict = pickle.load(file)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(filepath):
    #load Cifar-10 dataset
    _load_cifar_data(filepath)
    XList = []
    YList = []
    for number in range(1, 6):
        file = os.path.join(filepath, 'cifar-10-batches-py/data_batch_%d' % (number, ))
        X, Y = load_CIFAR10_batch(file)
        XList.append(X)
        YList.append(Y)
    train_x = np.concatenate(XList)
    train_y = np.concatenate(YList)
    del X, Y
    test_x, test_y = load_CIFAR10_batch(os.path.join(filepath, 'cifar-10-batches-py/test_batch'))
    return train_x, train_y, test_x, test_y

def _load_mnist_data(data_dir):
    if not os.path.isdir('Datasets'):
        os.system('mkdir Datasets/')
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    labelflag=False
    for target in ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']:
        if (not os.path.exists(target + '.pkl')):
            os.system('wget http://yann.lecun.com/exdb/mnist/' + target + '.gz')
            os.system('gunzip ' + target + '.gz')
            os.system('rm ' + target + '.gz')
            print 'Done Download'
            if labelflag:
                get_mnist_label(target)
            else:
                get_mnist_batch(target)
            print 'Done Extraction ' + target
            os.system('rm ' + target)
        else:
            print target + ' exits'
        labelflag = not labelflag
    os.chdir('..')
    os.chdir('..')

def get_mnist_batch(filename):
    binfile = open(filename,'rb')
    buf = binfile.read()
    binfile.close()
    pointer = 0
    num_magic, num_img, num_row, num_col = struct.unpack_from('>IIII', buf, pointer)
    pointer += struct.calcsize('IIII')
    imgs = np.zeros((1,28,28))
    for i in range(num_img):
        #The image size is 28 x 28 = 784
        img = np.array(struct.unpack_from('>784B', buf, pointer)).reshape((1,28,28))
        img[img > 1] = 1
        imgs = np.vstack((imgs,img))
        pointer += struct.calcsize('>784B')
    imgs[1:].dump(filename + '.pkl')

def get_mnist_label(filename):
    binfile = open(filename,'rb')
    buf = binfile.read()
    binfile.close()
    pointer = 0
    num_magic, num_label = struct.unpack_from('>II', buf, pointer)
    pointer += struct.calcsize('>II')
    labels = np.array([0])
    for l in range(num_label):
        label = np.array(struct.unpack_from('>1B', buf, pointer))
        pointer += struct.calcsize('>1B')
        labels = np.vstack((labels,label[0]))
    labels[1:].dump(filename + '.pkl')

def load_MNIST(filepath):
    _load_mnist_data(filepath)
    train_x = pickle.load(open(os.path.join(filepath,'train-images-idx3-ubyte.pkl'),'rb')) 
    train_y = pickle.load(open(os.path.join(filepath,'train-labels-idx1-ubyte.pkl'),'rb'))
    test_x = pickle.load(open(os.path.join(filepath,'t10k-images-idx3-ubyte.pkl'),'rb'))
    test_y = pickle.load(open(os.path.join(filepath,'t10k-labels-idx1-ubyte.pkl'),'rb'))
    return train_x, train_y, test_x, test_y
