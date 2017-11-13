import os
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys

from hybrid_data_provider import data_provider
from tensorflow.contrib import learn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# Parameters
learning_rate = float(sys.argv[8])
my_droup_out = float(sys.argv[9])
batch_size = 200
number_of_post_per_user = int(sys.argv[2])
train_iteration = int(sys.argv[3])
n_word_embedding = int(sys.argv[5])
n_letter_embedding = int(sys.argv[7])

# Network Parameters
n_sent_words = 100 
n_sent_letters = 600 
n_word_hidden = int(sys.argv[4]) # hidden layer num of features
n_letter_hidden = int(sys.argv[6])
n_classes = int(sys.argv[1])

dp = data_provider(size=n_classes, sent_max_len = n_sent_words, sent_max_char_len = n_sent_letters, number_of_post_per_user = number_of_post_per_user)

# tf Graph input
x = tf.placeholder(tf.int32, [None, n_sent_words])
u = tf.placeholder(tf.int32, [None, n_sent_letters])

y = tf.placeholder(tf.float32, [None, n_classes])

word_dropout = tf.placeholder(tf.float32, shape=())

is_training = tf.placeholder(tf.bool)

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([2*n_word_hidden + n_letter_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, u, weights, biases, dropout, is_training):

    with tf.variable_scope("word"):
        x = tf.cond(tf.equal(is_training, tf.constant(True)), lambda: tf.nn.dropout(x, dropout), lambda:x)
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_sent_words)
        x = tf.unstack(x, n_sent_words, 1)
        # Define a lstm cell with tensorflow
        fw_lstm_cell = rnn.BasicLSTMCell(n_word_hidden)
        bw_lstm_cell = rnn.BasicLSTMCell(n_word_hidden)

        outputs, states, _= rnn.static_bidirectional_rnn(fw_lstm_cell, bw_lstm_cell, x, dtype=tf.float32)

    with tf.variable_scope("letter"):
        u = tf.reshape(u, shape=[-1, n_sent_letters, n_letter_embedding, 1])

        conv1 = tf.layers.conv2d(u, n_letter_hidden, (4, n_letter_embedding), activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(u, n_letter_hidden, (3, n_letter_embedding), activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(u, n_letter_hidden, (2, n_letter_embedding), activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(u, n_letter_hidden, (5, n_letter_embedding), activation=tf.nn.relu)

        conv1 = tf.layers.max_pooling2d(conv1, strides=1, pool_size=(597, 1))
        conv2 = tf.layers.max_pooling2d(conv2, strides=1, pool_size=(598, 1))
        conv3 = tf.layers.max_pooling2d(conv3, strides=1, pool_size=(599, 1))
        conv4 = tf.layers.max_pooling2d(conv4, strides=1, pool_size=(596, 1))
        
        conv_final = tf.layers.max_pooling2d(tf.concat([conv1, conv2, conv3, conv4], 1), strides=1, pool_size = (4, 1))
        
        output_letter = tf.contrib.layers.flatten(conv_final)
        output_letter = tf.layers.dropout(output_letter, rate=dropout, training=is_training)
    
    max_output = outputs[0]

    for i in range(1, len(outputs)):
        max_output = tf.maximum(max_output, outputs[i])
   
    final_output = tf.concat((max_output, output_letter), 1)

    
    # Linear activation, using rnn inner loop last output
    return tf.matmul(final_output, weights['out']) + biases['out']


with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [dp.vocab_size, n_word_embedding], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x)

        letter_embedding = tf.get_variable("letter_embedding", [dp.vocab_char_size, n_letter_embedding], dtype=tf.float32)
        inputs_letter = tf.nn.embedding_lookup(letter_embedding, u)


pred = RNN(inputs, inputs_letter, weights, biases, word_dropout, is_training)

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
            batch_x, batch_y, batch_char_u= dp.get_next_train_batch(batch_size)
            
            acc, loss, _ = sess.run([accuracy, cost, optimizer], feed_dict={x: batch_x, y: batch_y, u: batch_char_u, word_dropout: my_droup_out, is_training: True})
            train_accr += acc 
            train_cost += loss
            
            step += 1
        
        lst_train_cost.append(train_cost/epoch_size)
        lst_train_accr.append(train_accr/epoch_size)
        
        
        valid_data, valid_label, valid_char_data = dp.get_next_valid_batch(dp.valid_size)

        acc , loss= sess.run([accuracy, cost], feed_dict={x: valid_data, y: valid_label, u:valid_char_data, word_dropout: 1.0, is_training:False})
        
        lst_valid_cost.append(loss)
        lst_valid_accr.append(acc)

        print(str(i) + "-TrainLoss = {:.3f}".format(train_cost/epoch_size) + ", TrainAccr= {:.3f}".format(train_accr/epoch_size) + ", ValidLoss = {:.3f}".format(loss) + ", ValidAccr= {:.3f}".format(acc))
        
    
    accr = 0
    accr_per_post = 0

    number_of_post = 0
    for i in range(n_classes):
        #print('for class number {0}'.format(i))
        test_data, test_label, test_char_data= dp.get_next_test_batch(i)
        loss, acc, prediction = sess.run([cost, accuracy, softmax_pred], feed_dict={x: test_data, y: test_label, u:test_char_data, word_dropout: 1.0, is_training:False})

        for predict in prediction:
            number_of_post += 1
            if predict.argmax(axis=0) == i:
                accr_per_post += 1

        result = np.sum(np.log10(prediction), axis=0)
        max_idx = result.argmax(axis=0)
        if max_idx == i :
            accr += 1
        

        #result = (np.sum(prediction, axis=0)/np.sum(np.sum(prediction, axis=0))).tolist()
        #temp = result[i]
        #result.sort(reverse=True)
        #max_index = result.index(temp)
        #print('  '.join([str(k) for k in result[:(max_index+1)]]))
        #print("Test Loss = {:.3f}".format(loss) + ", Test Accuracy= {:.3f}".format(acc))
    
    print('accr is {0:.3f} accr per post is {1:.3f}'.format(accr / n_classes, accr_per_post/number_of_post))

    #plt.plot(range(len(lst_train_cost)), lst_train_cost, 'g--', range(len(lst_valid_cost)), lst_valid_cost, 'b--')
    #plt.figure()

    #plt.plot(range(len(lst_train_accr)), lst_train_accr, 'g--', range(len(lst_valid_accr)), lst_valid_accr, 'b--')
    #plt.show()
