'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from data_provider import data_provider
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
dp = data_provider()

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 10

# Network Parameters
n_input = 30 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)
vocab_size = 18000
# tf Graph input
x = tf.placeholder(tf.int32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_input, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [vocab_size, n_hidden], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x)

pred = RNN(inputs, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Keep training until reach max iterations
    for i in range(1):
        print('epoch {0} :'.format(i))
        train_accr = 0.0
        valid_accr = 0.0
        train_cost = 0.0
        valid_cost = 0.0

        step = 0
        epoch_size = dp.train_size // batch_size
        while step < epoch_size:
            batch_x, batch_y = dp.get_next_train_batch(batch_size)

            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            train_accr += acc
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            train_cost += loss
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            step += 1
        

        print("Loss = {:.3f}".format(train_cost/epoch_size) + ", Training Accuracy= {:.3f}".format(train_accr/epoch_size))
        
        valid_data, valid_label = dp.get_next_valid_batch(100)

        acc = sess.run(accuracy, feed_dict={x: valid_data, y: valid_label})
        loss = sess.run(cost, feed_dict={x: valid_data, y: valid_label})
        
        print("Loss = {:.3f}".format(loss) + ", Training Accuracy= {:.3f}".format(acc))
        
    # Calculate accuracy for 128 mnist test images
    #test_len = 128
    
