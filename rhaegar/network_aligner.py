import csv
from schema import *
import refine
import numpy as np
import data_provider
import tensorflow as tf
import rnn as lstm
import matplotlib.pyplot as plt


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    accrs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
        "accuracy": model.accuracy
    }

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]
        accr = vals["accuracy"]

        costs += cost
        accrs += accr
        #iters += model.input.num_steps
        
    return costs / model.input.epoch_size, accrs / model.input.epoch_size



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

train_data = []
y_train_data = []
valid_data = []
y_valid_data = []

USER_COUNT = 10
MAX_SENT_LENGTH = 30

for twitter_username in lst_twitter_username[:USER_COUNT]:
    temp_set = []
    with open('../tumblr_twitter_scrapper/merged_tweets/{0}.csv'.format(twitter_username), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            item = TwitterItem(row)
            if item.is_owner == True:
                content = refine.clean(item.content, ignore_url= False)
                sents = refine.get_sentences(content)
                for i in range(len(sents)):
                    sents[i] = refine.stem(sents[i])
                content = ' <eos> '.join(sents)
                #if len(content.split(' ')) > 2:
                temp_set.append(content)

        random_set = np.zeros(len(temp_set))
        train_count = int(np.round(len(temp_set) * 0.7))
        random_set[:train_count] = 1
        np.random.shuffle(random_set)

        for i in range(len(temp_set)):
            if random_set[i] == 1:
                train_data.append(temp_set[i])
                k = lst_twitter_username.index(twitter_username)
                label = [0 for x in range(USER_COUNT)]
                label[k] = 1
                y_train_data.append(label)
            else:
                valid_data.append(temp_set[i])
                k = lst_twitter_username.index(twitter_username)
                label = [0 for x in range(USER_COUNT)]
                label[k] = 1
                y_valid_data.append(label)
test_data = []
y_test_data = []

for tumblr_username in lst_tumblr_username[:USER_COUNT]:
    with open('../tumblr_twitter_scrapper/merged_posts/{0}.csv'.format(tumblr_username), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            item = TumblrItem(row)
            if item.is_owner:
                content = refine.clean(item.content, ignore_url= False)
                for sent in refine.get_sentences(content):
                    if len(sent.split(' ')) > 2:
                        test_data.append(refine.stem(sent))
                        k = lst_tumblr_username.index(tumblr_username)
                        label = [0 for x in range(USER_COUNT)]
                        label[k] = 1
                        y_test_data.append(label)

word_2_id = data_provider.build_vocab(' '.join(train_data))

max_tweet_len = 0
max_tweet = ''
lst_len = []


train_set = []
valid_set = []
test_set = []


for txt in train_data:
    temp = data_provider.pad_word_ids(data_provider.text_to_word_ids(txt, word_2_id), MAX_SENT_LENGTH)
    if np.sum(temp) > 0:
        train_set.append(temp)


for txt in valid_data:
    temp = data_provider.pad_word_ids(data_provider.text_to_word_ids(txt, word_2_id), MAX_SENT_LENGTH)
    if np.sum(temp) > 0:
        valid_set.append(temp)


for txt in test_data:
    temp = data_provider.pad_word_ids(data_provider.text_to_word_ids(txt, word_2_id), MAX_SENT_LENGTH)
    if np.sum(temp) > 0:
        test_set.append(temp)


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
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope("Train"):
        train_input = lstm.LSTMInput(
            config=config, data=train_set, y_data=y_train_data, number_of_class= USER_COUNT, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = lstm.LSTMNetwork(is_training=True, config=config, input_=train_input)
        
    with tf.name_scope("Valid"):
        valid_input = lstm.LSTMInput(
            config=config, data=train_set, y_data= y_train_data, number_of_class= USER_COUNT, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = lstm.LSTMNetwork(is_training=False, config=config, input_=valid_input)
    

    sv = tf.train.Supervisor()
    with sv.managed_session() as session:
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)   
            #lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            #lr = config.learning_rate * lr_decay
            #m.assign_lr(lr, session)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_cost, trian_accr = run_epoch(session, m, eval_op=m.train_op, verbose=True)

            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_cost))
            print("Epoch: %d Train accr: %.3f"%(i+1, trian_accr))

            valid_cost, valid_accr = run_epoch(session, mvalid)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_cost))
            print("Epoch: %d Train accr: %.3f"%(i+1, valid_accr))

