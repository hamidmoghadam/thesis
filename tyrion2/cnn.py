'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys
# Import MNIST data
from data_provider import data_provider
from tensorflow.contrib import learn
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

def miror_data(x, y):
     x = np.concatenate((np.array(x), np.fliplr(x)), axis=0)
     y = np.concatenate((np.array(y), np.array(y)), axis=0)
     return x, y

# Parameters
learning_rate = float(sys.argv[5])#0.0007
batch_size = 200
number_of_post_per_user = int(sys.argv[2])
train_iteration = int(sys.argv[3])
n_fully_connected = int(sys.argv[6])
# Network Parameters
n_input = 600 # MNIST data input (img shape: 28*28)
n_hidden = int(sys.argv[4]) # hidden layer num of features
n_classes = int(sys.argv[1]) # MNIST total classes (0-9 digits)

#vocab_size = 58000
dp = data_provider(size=n_classes, sent_max_char_len = n_input, number_of_post_per_user = number_of_post_per_user)

# tf Graph input
x = tf.placeholder(tf.int32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
dropout = tf.placeholder(tf.float32, shape=())
is_training = tf.placeholder(tf.bool)

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def conv_net(x, n_classes, dropout, is_training):
    x = tf.reshape(x, shape=[-1, n_input, 1, n_hidden])
    print(x)
    #batch (sentsie * n_hidden|embedding)
    # Convolution Layer with 32 filters and a kernel size of 5
    conv1 = tf.layers.conv2d(x, 32, (4, 1), activation=tf.nn.relu, padding='same')
    print(conv1)
    # Max Pooling (down-sampling) with strides of 1 and kernel size of 2
    conv1 = tf.layers.max_pooling2d(conv1, strides = 2, pool_size= (2,1))
    print(conv1)
    # Convolution Layer with 64 filters and a kernel size of 3
    conv2 = tf.layers.conv2d(conv1, 64, (2,1), activation=tf.nn.relu, padding='same')
    print(conv2)
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    conv2 = tf.layers.max_pooling2d(conv2, strides= 2, pool_size = (2, 1))
    print(conv2)

    
    fc1 = tf.reduce_max(conv2, [3])

    fc1 = tf.contrib.layers.flatten(fc1)
    
    
    # Flatten the data to a 1-D vector for the fully connected layer
    #fc1 = tf.contrib.layers.flatten(conv2)
    
    # Fully connected layer (in tf contrib folder for now)
    #fc1 = tf.layers.dense(fc1, n_fully_connected)

    # Apply Dropout (if is_training is False, dropout is not applied)
    fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

    # Output layer, class prediction
    out = tf.layers.dense(fc1, n_classes)

    return out

def RNN(x, weights, biases, dropout, is_training):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    x = tf.cond(tf.equal(is_training, tf.constant(True)), lambda: tf.nn.dropout(x, dropout), lambda:x)
    #if is_training:
    #    x = tf.nn.dropout(x, dropout)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_input, 1)

    # Define a lstm cell with tensorflow
    fw_lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    bw_lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
	
    # Get lstm cell output
    #outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    outputs, states, _ = rnn.static_bidirectional_rnn(fw_lstm_cell, bw_lstm_cell, x, dtype=tf.float32)


    output = outputs[0]
    for i in range(1, len(outputs)):
        output = tf.maximum(output, outputs[i])
    # Linear activation, using rnn inner loop last output
    return tf.matmul(output, weights['out']) + biases['out']

with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [dp.letter_dic_size, n_hidden], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x)

#pred = RNN(inputs, weights, biases, dropout, is_training)
pred = conv_net(inputs, n_classes, dropout, is_training)
softmax_pred = tf.nn.softmax(pred)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

lst_train_cost = []
lst_valid_cost = []

lst_train_accr = []
lst_valid_accr = []
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    #sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
    
    # Keep training until reach max iterations
    for i in range(train_iteration):
        #print('epoch {0} :'.format(i+1))
        train_accr = 0.0
        valid_accr = 0.0
        train_cost = 0.0
        valid_cost = 0.0

        step = 0
        epoch_size = max(dp.train_size // batch_size, 1)
        while step < epoch_size:
            batch_x, batch_y, batch_char_x = dp.get_next_train_batch(batch_size)

            batch_char_x = np.concatenate((np.array(batch_char_x), np.fliplr(batch_char_x)), axis=0)
            batch_y = np.concatenate((np.array(batch_y), np.array(batch_y)), axis=0)

            miror_data(batch_char_x, batch_y)
            
            acc, loss, _ = sess.run([accuracy, cost, optimizer], feed_dict={x: batch_char_x , y: batch_y, dropout: 0.5, is_training: True})
            train_accr += acc 
            train_cost += loss
            
            step += 1
        
        lst_train_cost.append(train_cost/epoch_size)
        lst_train_accr.append(train_accr/epoch_size)
        #'''
        print("Training Loss = {:.3f}".format(train_cost/epoch_size) + ", Training Accuracy= {:.3f}".format(train_accr/epoch_size))
        
        valid_data, valid_label, valid_char = dp.get_next_valid_batch(dp.valid_size)
        miror_data(valid_char, valid_label)
        #print(np.array(valid_char).shape)
        acc, loss = sess.run([accuracy, cost], feed_dict={x: valid_char, y: valid_label, dropout: 1.0, is_training:False})
        lst_valid_cost.append(loss)
        lst_valid_accr.append(acc)
        
        print("Validation Loss = {:.3f}".format(loss) + ", Validation Accuracy= {:.3f}".format(acc))
        #'''
    
    accr = 0
    accr_per_post = 0

    number_of_post = 0
    for i in range(n_classes):
        #print('for class number {0}'.format(i))
        test_data, test_label, test_char = dp.get_next_test_batch(i)
        loss, acc, prediction = sess.run([cost, accuracy, softmax_pred], feed_dict={x: test_char, y: test_label, dropout: 1.0, is_training:False})

        for predict in prediction:
            number_of_post += 1
            if predict.argmax(axis=0) == i:
                accr_per_post += 1

        result = np.sum(np.log10(prediction), axis=0)
        max_idx = result.argmax(axis=0)
        if max_idx == i :
            accr += 1
        
    
    print('accr is {0:.3f} accr per post is {1:.3f}'.format(accr / n_classes, accr_per_post/number_of_post))

    #plt.plot(range(len(lst_train_cost)), lst_train_cost, 'g--', range(len(lst_valid_cost)), lst_valid_cost, 'b--')
    #plt.figure()

    #plt.plot(range(len(lst_train_accr)), lst_train_accr, 'g--', range(len(lst_valid_accr)), lst_valid_accr, 'b--')
    #plt.show()

