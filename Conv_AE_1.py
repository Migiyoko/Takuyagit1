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


n_sample_train = 50000
n_sample_test = 10000


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

    mnist = input_data.read_data_sets('./Data', one_hot=True)
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


def conv2d(input, name, kshape, strides=[1, 1, 1, 1]):
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

def deconv2d(input, name, kshape, n_outputs, strides=[1, 1], activation=tf.nn.relu):
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


train_x, one_hots_train, test_x, one_hots_test,mnist = get_MNIST_data()
number_test = [one_hots_test[i, :].argmax() for i in range(0, one_hots_test.shape[0])]

train_x=np.reshape(train_x,(-1,28,28,1))
test_x=np.reshape(test_x,(-1,28,28,1))

plot_MNIST(x=train_x, one_hot=one_hots_train)

n_label = len(np.unique(number_test))   # Number of class
height = train_x.shape[1]               # All the pixels are represented as a vector (dim: 784)

z_dimension = 2 # Latent space dimension

# Hyperparameters
hyperparameters_encode= {'en_filter':[[5,5,1,8],[3,3,8,16],[2,2,16,32]],
                          'en_size':[784,256,64,z_dimension],
                          'en_activation':['relu','relu','linear'],
                          'names':['en_layer_1', 'en_layer_2', 'latent_space']}
hyperparameters_decode= {'de_filter':[[3,3],[3,3],[3,3]],
                         'de_size':[z_dimension,64,256,784],
                          'de_activation':['relu','relu','relu'],
                          'names':['de_layer_1', 'de_layer_2', 'de_layer_out']}
hyperparameters_scope={'learning_late':0.001, 'maxEpoch':200, 'batch_size':512}

# Session and context manager
tf.reset_default_graph()
sess = tf.Session()
input_img = Input(shape=(1, 28, 28))
with tf.variable_scope(tf.get_variable_scope()):

    # Placeholders
    x = tf.placeholder(tf.float32, [None, 28,28,1], name='x')
    dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

    # Model
    print('')
    print("ENCODER")
    print(x)

    c1=conv2d(x, name='c1', kshape=hyperparameters_encode['en_filter'][0])
    p1 = maxpool2d(c1, name='p1')
    do1 = dropout(p1, name='do1',keep_prob=0.9)
    c2=conv2d(do1, name='c2', kshape=hyperparameters_encode['en_filter'][1])
    p2 = maxpool2d(c2, name='p2')
    do2 = dropout(p2, name='do2', keep_prob=0.9)

    print(do2)

    flatten= tf.reshape(do2,[-1,784])

    print(flatten)

    l1 = dense(flatten, in_size=hyperparameters_encode['en_size'][0],
               out_size=hyperparameters_encode['en_size'][1],
               activation=hyperparameters_encode['en_activation'][0]
               , name=hyperparameters_encode['names'][0])

    print(l1)
    l2 = dense(l1, in_size=hyperparameters_encode['en_size'][1],
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
    l4 = dense(z, in_size=hyperparameters_decode['de_size'][0],
               out_size=hyperparameters_decode['de_size'][1],
               activation=hyperparameters_decode['de_activation'][0]
               , name=hyperparameters_decode['names'][0])

    print(l4)
    l5 = dense(l4, in_size=hyperparameters_decode['de_size'][1],
               out_size=hyperparameters_decode['de_size'][2],
               activation=hyperparameters_decode['de_activation'][1]
               , name=hyperparameters_decode['names'][1])

    print(l5)
    decoded_input=dense(l5, in_size=hyperparameters_decode['de_size'][2],
               out_size=hyperparameters_decode['de_size'][3],
               activation=hyperparameters_decode['de_activation'][2]
               , name=hyperparameters_decode['names'][2])
    decoded_input=tf.reshape(decoded_input,[-1,7,7,16])


    up2 = upsample(decoded_input, name='up2', factor=[2, 2])
    do5 = dropout(up2, name='do5', keep_prob=0.9)
    dc2 = deconv2d(do5, name='dc2', kshape=hyperparameters_decode['de_filter'][1], n_outputs=8)
    up3 = upsample(dc2, name='up3', factor=[2, 2])
    do6 = dropout(up3, name='do6', keep_prob=0.9)
    x_hat = deconv2d(do6, name='x_hat', kshape=hyperparameters_decode['de_filter'][2], n_outputs=1)

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
    saver = tf.train.Saver()
    saver.save(sess, '../MNIST-TensorFlow-master/model/model_1')
    print('saved')
    plt.figure()
    plt.plot(loss_history)




    # Encode the training data
    train_data = {x: train_x}
    encoded_train = sess.run(z, feed_dict=train_data)
    test_data = {x: test_x}
    encoded_test = sess.run(z, feed_dict=test_data)
    number_train = [one_hots_train[i, :].argmax() for i in range(0, one_hots_train.shape[0])]
    latent_space_plot(encoded_train, one_hots_train, encoded_test, one_hots_test)

    reconstructed=sess.run(x_hat,feed_dict=train_data)
    reconstructed_test = sess.run(x_hat, feed_dict=test_data)

    #KNN
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
    print('Test accuracy is:' + str(accuracy))

    #figure of law data and reconstructed data
    plt.figure()
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.title(np.argmax(one_hots_train, axis=1)[i])
        plt.axis("off")
        plt.imshow(train_x[i].reshape(28, 28), cmap=cm.gray_r)

    plt.figure()
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        # plt.title(label_test_predicted[i])
        plt.axis("off")
        plt.imshow(reconstructed[i].reshape(28, 28), cmap=cm.gray_r)

    plt.figure()
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.title(label_test_true[i])
        plt.axis("off")
        plt.imshow(test_x[i].reshape(28, 28), cmap=cm.gray_r)

    plt.figure()
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.title(label_test_predicted[i])
        plt.axis("off")
        plt.imshow(reconstructed_test[i].reshape(28, 28), cmap=cm.gray_r)

