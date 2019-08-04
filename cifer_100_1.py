from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from keras.layers import Convolution2D, Input,UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import time
import pandas as pd
import os
import _pickle as cPickle
# import sys, codecs
# import locale
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout)
# locale.setlocale(locale.LC_CTYPE, ('UTF-8'))
from keras.utils import np_utils


n_sample_train = 50000
n_sample_test = 10000
nb_classes=100


def latent_space_plot(encoded_train, one_hots_train,encoded_test,one_hots_test):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    cmap = cm.get_cmap('jet', one_hots_train.shape[1])
    colors = []
    labels_train = np.argmax(one_hots_train, axis=1)
    labels_test = np.argmax(one_hots_test, axis=1)
    for i in range(cmap.N):
        colors.append(cmap(i))
    for i in range(one_hots_train.shape[1]):
        ind = np.where(labels_train == i)
        ax.scatter(encoded_train[ind, 0], encoded_train[ind, 1],
                   color=colors[i], s=5, label='train_' + str(i))
        plt.legend()
    for i in range(one_hots_test.shape[1]):
        ind = np.where(labels_test == i)
        ax.scatter(encoded_test[ind, 0], encoded_test[ind, 1],
                   color=colors[i], s=100, label='test_' + str(i))
        plt.legend()
def get_MNIST_data():

    mnist = input_data.read_data_sets('./Data1', one_hot=True)
    train_x, one_hots_train = mnist.train.next_batch(n_sample_train)
    test_x, one_hots_test = mnist.train.next_batch(n_sample_test)

    return train_x, one_hots_train, test_x, one_hots_test,mnist

def plot_MNIST(x, one_hot):

    row = 4
    column = 4
    p = random.sample(range(1, 100), row * column)

    plt.figure()

    for i in range(row * column):

        image = x[p[i]].reshape(28, 28)
        plt.subplot(row, column, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title('label = {}'.format(np.argmax(one_hot[p[i]]).astype(int)))
        plt.axis('off')

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                    wspace=0.05, hspace=0.3)
    plt.show()

def dense(inputs, in_size, out_size, activation='sigmoid', name='layer'):

    with tf.variable_scope(name, reuse=False):

        w = tf.get_variable("w", shape=[in_size, out_size], initializer=tf.random_normal_initializer(mean=0., stddev=0.1))
        b = tf.get_variable("b", shape=[out_size], initializer=tf.constant_initializer(0.0))

        l = tf.add(tf.matmul(inputs, w), b)

        if activation == 'relu':
            l = tf.nn.relu(l)
        elif activation == 'sigmoid':
            l = tf.nn.sigmoid(l)
        elif activation == 'tanh':
            l = tf.nn.tanh(l)
        elif activation == 'leaky_relu':
            l = tf.nn.leaky_relu(l)
        else:
            l = l

        # l = tf.nn.dropout(l, rate=dropout_rate)

    return l


def conv2d(input, name, kshape, strides=[1, 2, 2, 1]):
    with tf.name_scope(name):
        W = tf.get_variable(name='w_'+name,
                            shape=kshape)
        b = tf.get_variable(name='b_' + name,
                            shape=[kshape[3]])
        out = tf.nn.conv2d(input,W,strides=strides, padding='SAME')
        out = tf.nn.bias_add(out, b)
        out = tf.nn.relu(out)
        return out
    #filter=[height, width, in_channel, out_channel],strides=[1, up_down_stride, side_stride,1]

def maxpool2d(x,name,kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.name_scope(name):
        out = tf.nn.max_pool(x,
                             ksize=kshape,
                             strides=strides,
                             padding='SAME')
        return out

def deconv2d(input, name, kshape, n_outputs, strides=[2, 2], activation=tf.nn.relu):
    with tf.name_scope(name):
        out = tf.contrib.layers.conv2d_transpose(input,
                                                 num_outputs= n_outputs,
                                                 kernel_size=kshape,
                                                 stride=strides,
                                                 padding='SAME',
                                                 activation_fn=activation)
        return out


def upsample(input, name, factor=[2,2]):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
        return out

def dropout(input, name, keep_prob):
    with tf.name_scope(name):
        out = tf.nn.dropout(input, keep_prob)
        return out

def unpickle(file):
    with open(file, 'rb') as f:
        fo = f.read()
    # fo = open(file, 'rb')
    # with cPickle.load(fo, "r", "Shift-JIS", "ignore") as file:
    #     df = pd.read_table(file, delimiter=",")
    #     print(df)
    dict = cPickle.loads(fo,encoding='latin1')
    # fo.close()
    return dict

def get_cifar100(folder):
    train_fname = os.path.join(folder,'train')
    test_fname  = os.path.join(folder,'test')
    data_dict = unpickle(train_fname)
    train_data = data_dict['data']
    train_fine_labels = data_dict['fine_labels']
    train_coarse_labels = data_dict['coarse_labels']

    data_dict = unpickle(test_fname)
    test_data = data_dict['data']
    test_fine_labels = data_dict['fine_labels']
    test_coarse_labels = data_dict['coarse_labels']

    bm = unpickle(os.path.join(folder, 'meta'))
    clabel_names = bm['coarse_label_names']
    flabel_names = bm['fine_label_names']

    return train_data, np.array(train_coarse_labels), np.array(train_fine_labels), test_data, np.array(test_coarse_labels), np.array(test_fine_labels), clabel_names, flabel_names

tr_data100, tr_clabels100, tr_flabels100, te_data100, te_clabels100, te_flabels100, clabel_names100, flabel_names100 = get_cifar100('./Data2/cifar-100-python')
# train_x, one_hots_train, test_x, one_hots_test,mnist = get_MNIST_data()
number_test = te_flabels100

train_x=np.reshape(tr_data100,(-1,32,32,3))
test_x=np.reshape(te_data100,(-1,32,32,3))
one_hots_train = np_utils.to_categorical(tr_flabels100, nb_classes)
one_hots_test = np_utils.to_categorical(te_flabels100, nb_classes)


# plot_MNIST(x=train_x, one_hot=one_hots_train)

n_label = 100   # Number of class
# height = train_x.shape[1]               # All the pixels are represented as a vector (dim: 784)

z_dimension = 2 # Latent space dimension

# Hyperparameters
hyperparameters_encode= {'en_filter':[[3,3,3,16],[3,3,16,32],[3,3,32,64],[3,3,64,128]],
                         'en_size':[512,256,128,z_dimension],
                         'en_activation':['relu','relu','linear'],
                         'names':['en_layer_1', 'en_layer_2', 'latent_space']}
hyperparameters_decode= {'de_filter':[[3,3],[3,3],[3,3],[3,3]],
                         'de_size':[z_dimension,128,256,512],
                         'de_activation':['relu','relu','relu'],
                         'names':['de_layer_1', 'de_layer_2', 'de_layer_out']}
hyperparameters_scope={'learning_late':0.0001, 'maxEpoch':100, 'batch_size':512}

# Session and context manager
tf.reset_default_graph()
sess = tf.Session()
# input_img = Input(shape=(1, 28, 28))
with tf.variable_scope(tf.get_variable_scope()):

    # Placeholders
    x = tf.placeholder(tf.float32, [None, 32,32,3], name='x')
    dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

    # Encoder
    print('')
    print("ENCODER")
    print(x)

    c1=conv2d(x, name='c1', kshape=hyperparameters_encode['en_filter'][0])
    # p1 = maxpool2d(c1, name='p1')
    do1 = dropout(c1, name='do1',keep_prob=1.0)
    c2=conv2d(do1, name='c2', kshape=hyperparameters_encode['en_filter'][1])
    # p2 = maxpool2d(c2, name='p2')
    do2 = dropout(c2, name='do2', keep_prob=1.0)
    c3=conv2d(do2, name='c3', kshape=hyperparameters_encode['en_filter'][2])
    # p3 = maxpool2d(c3, name='p3')
    do3 = dropout(c3, name='do3', keep_prob=1.0)
    c4=conv2d(do3, name='c4', kshape=hyperparameters_encode['en_filter'][3])
    # p3 = maxpool2d(c3, name='p3')
    do4 = dropout(c4, name='do4', keep_prob=1.0)

    # c=conv_encode(x)
    print(do4)

    flatten= tf.reshape(do4,[-1,512])

    print(flatten)

    # l1 = dense(flatten, in_size=hyperparameters_encode['en_size'][0],
    #            out_size=hyperparameters_encode['en_size'][1],
    #            activation=hyperparameters_encode['en_activation'][0],
    #            name=hyperparameters_encode['names'][0])

    # print(l1)
    l2 = dense(flatten, in_size=hyperparameters_encode['en_size'][0],
               out_size=hyperparameters_encode['en_size'][2],
               activation=hyperparameters_encode['en_activation'][1]
               , name=hyperparameters_encode['names'][1])

    print(l2)
    z=dense(l2, in_size=hyperparameters_encode['en_size'][2],
               out_size=hyperparameters_encode['en_size'][3],
               activation=hyperparameters_encode['en_activation'][2]
               , name=hyperparameters_encode['names'][2])
    print(z)

    print("DECODER")
    # Decoder
    # l4 = dense(z, in_size=hyperparameters_decode['de_size'][0],
    #            out_size=hyperparameters_decode['de_size'][1],
    #            activation=hyperparameters_decode['de_activation'][0]
    #            , name=hyperparameters_decode['names'][0])

    # print(l4)
    l5 = dense(z, in_size=hyperparameters_decode['de_size'][0],
               out_size=hyperparameters_decode['de_size'][2],
               activation=hyperparameters_decode['de_activation'][1]
               , name=hyperparameters_decode['names'][1])

    print(l5)
    decoded_input=dense(l5, in_size=hyperparameters_decode['de_size'][2],
               out_size=hyperparameters_decode['de_size'][3],
               activation=hyperparameters_decode['de_activation'][2]
               , name=hyperparameters_decode['names'][2])
    decoded_input=tf.reshape(decoded_input,[-1,2,2,128])

    do5 = dropout(decoded_input, name='do5', keep_prob=1.0)
    # up1 = upsample(do4, name='up1', factor=[2, 2])
    dc1=deconv2d(do5,name='dc1',kshape=hyperparameters_decode['de_filter'][0],n_outputs=64)
    do6 = dropout(dc1, name='do6', keep_prob=1.0)
    # up1 = upsample(do4, name='up1', factor=[2, 2])
    dc2=deconv2d(do6,name='dc2',kshape=hyperparameters_decode['de_filter'][1],n_outputs=32)
    # up2 = upsample(decoded_input, name='up2', factor=[2, 2])
    do7 = dropout(dc2, name='do7', keep_prob=1.0)
    dc3 = deconv2d(do7, name='dc3', kshape=hyperparameters_decode['de_filter'][2], n_outputs=16)
    # up3 = upsample(dc2, name='up3', factor=[2, 2])
    do8 = dropout(dc3, name='do8', keep_prob=1.0)
    x_hat = deconv2d(do8, name='x_hat', kshape=hyperparameters_decode['de_filter'][3], n_outputs=3)

    # x_hat=conv_decode(encoded_input=tf.reshape(decoded_input,[-1,4,4,8]))

    print(x_hat)
    # Scope
    learning_rate=tf.Variable(hyperparameters_scope['learning_late'],
                              trainable=False)
    # Loss function
    loss=tf.reduce_mean(tf.square(x-x_hat))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer').minimize(loss)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='optimizer').minimize(loss)
    # tensorboard
    writer=tf.summary.FileWriter('.\Tensorboard') #tensorboard -- logdir=./Tensorboard
    writer.add_graph(graph=sess.graph)

    # Initialize the Neural Network
    sess.run(tf.global_variables_initializer())

    # Train the Neural Network
    loss_history=[]
    t1=time.time()

    for epoch in range(hyperparameters_scope['maxEpoch']):
        i=0
        loss_batch=[]
        while i<n_sample_train:
            start=i
            end=i + hyperparameters_scope['batch_size']
            train_data={x:train_x[start:end]}
            _, l=sess.run([optimizer, loss], feed_dict=train_data)
            loss_batch.append(l)
            i=i+hyperparameters_scope['batch_size']

        epoch_loss=np.mean(loss_batch)
        loss_history.append(epoch_loss)
        print('Epoch',epoch, '/', hyperparameters_scope['maxEpoch'], '. :Loss',epoch_loss)
    t2=time.time()

    print('learning_time=',t2-t1)
    plt.figure()
    plt.plot(loss_history)




    # Encode the training data
    # train_data={x:train_x}
    # encoded=sess.run(z,feed_dict=train_data)
    # plt.figure()
    # number_train = [one_hots_train[i, :].argmax() for i in range(0, one_hots_train.shape[0])]
    train_data = {x: train_x}
    encoded_train = sess.run(z, feed_dict=train_data)
    test_data = {x: test_x}
    encoded_test = sess.run(z, feed_dict=test_data)
    number_train = [one_hots_train[i, :].argmax() for i in range(0, one_hots_train.shape[0])]
    latent_space_plot(encoded_train, one_hots_train, encoded_test, one_hots_test)
    # for label in np.unique(number_train):
    #     plt.scatter(encoded[number_train==label, 0], encoded[number_train==label,1],s=5)
    # # plt.figure()
    # # plt.scatter(encoded[:, 0], encoded[:, 1], c=number_train,s=5)
    # label= np.unique(number_train)
    # plt.legend(label)
    # Reconstruct the data at the output of the decoder
    reconstructed=sess.run(x_hat,feed_dict=train_data)
    reconstructed_test = sess.run(x_hat, feed_dict=test_data)
    # plt.figure()
    # plt.scatter(reconstructed[:, 0], reconstructed[:,1])
    nn = 10
    neigh = KNeighborsClassifier(n_neighbors=nn)
    neigh.fit(encoded_train, np.argmax(one_hots_train, axis=1))
    label_test_predicted = neigh.predict(encoded_test)
    label_test_true = np.argmax(one_hots_test, axis=1)

    score = 0
    for i in range(one_hots_test.shape[0]):
        if label_test_predicted[i] == label_test_true[i]:
            score += 1

    accuracy = np.round(score / one_hots_test.shape[0], 4)
    print('Test accuracu is:' + str(accuracy))

    # plt.figure()
    # for i in range(25):
    #     plt.subplot(5, 5, i+1)
    #     plt.title(np.argmax(one_hots_train, axis=1)[i])
    #     plt.axis("off")
    #     # img = plt.imread(train_x[i].reshape(32, 32,3))
    #     img=train_x[i].reshape(32, 32,3)

    train_x=(train_x.reshape(50000, 3, 32, 32).transpose(0,2,3,1))/255
    reconstructed=(reconstructed.reshape(50000, 3, 32, 32).transpose(0,2,3,1))/255
    test_x=(test_x.reshape(10000,3,32,32).transpose(0,2,3,1))/255
    reconstructed_test = (reconstructed_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1))/255

    plt.figure()
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.title(flabel_names100[np.argmax(one_hots_train, axis=1)[i]])
        plt.axis("off")
        plt.imshow(train_x[i])

    plt.figure()
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        # plt.title(label_test_predicted[i])
        plt.axis("off")
        plt.imshow(reconstructed[i])

    plt.figure()
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.title(flabel_names100[label_test_true[i]])
        plt.axis("off")
        plt.imshow(test_x[i])

    plt.figure()
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.title(flabel_names100[label_test_predicted[i]])
        plt.axis("off")
        plt.imshow(reconstructed_test[i])




# Plot the latent space

# Plot reconstruction

# PCA
# plt.figure()
# plt.imshow(reconstructed_test[0])