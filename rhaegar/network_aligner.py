import csv
from schema import *
import refine
import numpy as np
import data_provider
import tensorflow as tf
import lstm
import matplotlib.pyplot as plt


lst_username = []
lst_twitter_username = []
lst_tumblr_username = []
with open('../tumblr_twitter_scrapper/large_username_pairs_filtered.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        lst_username.append(
            {'tumblr': row[0], 'twitter': row[2].replace(r'twitter.com/', '')})
        lst_twitter_username.append(row[2].replace(r'twitter.com/', ''))
        lst_tumblr_username.append(row[0])

train_set = []
y_train_set = []
valid_set = []
y_valid_set = []

USER_COUNT = 3
MAX_SENT_LENGTH = 40

for twitter_username in lst_twitter_username[:USER_COUNT]:
    temp_set = []
    with open('../tumblr_twitter_scrapper/tweets/{0}.csv'.format(twitter_username), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            item = TwitterItem(row)
            if item.is_owner == True:
                content = refine.clean(item.content)
                sents = refine.get_sentences(content)
                for i in range(len(sents)):
                    sents[i] = refine.stem(sents[i])
                content = ' <eos> '.join(sents)
                if len(content.split(' ')) > 2:
                    temp_set.append(content)

        random_set = np.zeros(len(temp_set))
        train_count = int(np.round(len(temp_set) * 0.7))
        random_set[:train_count] = 1
        np.random.shuffle(random_set)

        for i in range(len(temp_set)):
            if random_set[i] == 1:
                train_set.append(temp_set[i])
                k = lst_twitter_username.index(twitter_username)
                label = [0 for x in range(USER_COUNT)]
                label[k] = 1
                y_train_set.append(label)
            else:
                valid_set.append(temp_set[i])
                k = lst_twitter_username.index(twitter_username)
                label = [0 for x in range(USER_COUNT)]
                label[k] = 1
                y_valid_set.append(label)
test_set = []
y_test_set = []

for tumblr_username in lst_tumblr_username[:USER_COUNT]:
    with open('../tumblr_twitter_scrapper/posts/{0}.csv'.format(tumblr_username), 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            item = TumblrItem(row)
            if item.is_owner:
                content = refine.clean(item.content)
                for sent in refine.get_sentences(content):
                    if len(sent.split(' ')) > 2:
                        test_set.append(refine.stem(sent))
                        k = lst_tumblr_username.index(tumblr_username)
                        label = [0 for x in range(USER_COUNT)]
                        label[k] = 1
                        y_test_set.append(label)

word_2_id = data_provider.build_vocab(' '.join(train_set))

max_tweet_len = 0
max_tweet = ''
lst_len = []

for i in range(len(train_set)):
    train_set[i] = data_provider.pad_word_ids(data_provider.text_to_word_ids(train_set[i], word_2_id), MAX_SENT_LENGTH)
    lst_len.append(len(train_set[i]))


for i in range(len(valid_set)):
    valid_set[i] = data_provider.pad_word_ids(data_provider.text_to_word_ids(valid_set[i], word_2_id), MAX_SENT_LENGTH)
    lst_len.append(len(valid_set[i]))


for i in range(len(test_set)):
    test_set[i] = data_provider.pad_word_ids(data_provider.text_to_word_ids(test_set[i], word_2_id), MAX_SENT_LENGTH)
    lst_len.append(len(test_set[i]))


'''
t = 40

plt.hist([x for x in lst_len if x <= t])
plt.figure()
plt.boxplot([x for x in lst_len if x <= t])


print(max(lst_len))
print(len([x for x in lst_len if x > t]))
print(len(lst_len))


plt.show()
'''
config = lstm.BestConfig()
eval_config = lstm.BestConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-1 * config.init_scale, config.init_scale)

    with tf.name_scope("Train"):
        train_input = lstm.LSTMInput(
            config=config, data=train_set, y_data=y_train_set, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = lstm.LSTMNetwork(is_training=True, config=config, input=train_input)
    '''    
    with tf.name_scope("Valid"):
        valid_input = lstm.LSTMInput(
            config=config, data=valid_set, y_data= y_valid_set, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = lstm.LSTMNetwork(
                is_training=False, config=config, input=valid_input)'''
    

    sv = tf.train.Supervisor()
    with sv.managed_session() as session:
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            lr = config.learning_rate * lr_decay
            m.set_lr(lr, session)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, lr))

            train_cost = m.run_epoch(session)
            print("Epoch: %d Train cost: %.3f" %(i + 1, train_cost))

            '''            
            valid_cost = mvalid.run_epoch(session)
            print("Epoch: %d Valid Cost: %.3f" %
                        (i + 1, valid_cost))
        '''
