import tensorflow as tf
import numpy as np
from difflib import SequenceMatcher
import csv
from schema import *
import refine
import collections

class data_provider(object):
    def __init__(self):
       
        lst_username = []
        lst_twitter_username = []
        lst_tumblr_username = []

        with open('../tumblr_twitter_scrapper/large_username_pairs_filtered.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                lst_username.append({'tumblr': row[0], 'twitter': row[2].replace(r'twitter.com/', '')})
                lst_twitter_username.append(row[2].replace(r'twitter.com/', ''))
                lst_tumblr_username.append(row[0])

        data = []    
        
        for twitter_username in lst_twitter_username:
            with open('../tumblr_twitter_scrapper/merged_tweets/{0}.csv'.format(twitter_username), 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    item = TwitterItem(row)
                    if item.is_owner == True:
                        content = refine.clean(item.content, ignore_url= False, ignore_stopword=True)
                        data.append(content)

        self.Vocab = self.build_vocab(' '.join(data))
        self.VocabSize = len(self.Vocab)

    def build_vocab(self,data):
        return list(set(data.split(' ')))





dp = data_provider()

word_embedding_size = 5
vocab_size = 5000
train_iteration = 1000


print('vocab size is {0}'.format(dp.VocabSize))


x_data = tf.placeholder(tf.float32, [None, word_embedding_size])
y_data = tf.placeholder(tf.float32, [None, 1])


W = tf.Variable(tf.random_normal([word_embedding_size, word_embedding_size], stddev=0.5))
b = tf.Variable(tf.random_normal([word_embedding_size], stddev=0.5))


x =  tf.matmul(x_data, W) + b
temp = tf.Variable(tf.zeros([vocab_size, word_embedding_size]), tf.float32)
temp.assign(x)


print(x)

x_1 = tf.tile(x, [vocab_size, 1])
x_2 = tf.reshape(tf.tile(temp, [1, vocab_size]), [-1, word_embedding_size])

print(x_1)
print(x_2)


y_pred = tf.norm(x_1 - x_2, ord='euclidean', axis=1, keep_dims=True)
y_pred = tf.div(tf.subtract(y_pred, tf.reduce_min(y_pred)), tf.subtract(tf.reduce_max(y_pred), tf.reduce_min(y_pred)))

print(y_pred)
print(y_data)

cost = tf.reduce_mean(tf.square(y_pred - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_data))                 
#optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    x_set = np.random.uniform(low=0.0, high=1.0, size=(vocab_size, word_embedding_size)).astype(np.float32)
    y_set = []

    for i in range(vocab_size):
        for j in range(vocab_size):
            r = SequenceMatcher(None, dp.Vocab[i], dp.Vocab[j]).ratio()
            y_set.append(r)
    

    y_set = np.array(y_set).reshape([-1, 1])

    print('training started ...')
    for i in range(train_iteration):
        _, loss = sess.run([optimizer, cost], feed_dict={x_data: x_set, y_data: y_set})
        print(loss)
        
        


