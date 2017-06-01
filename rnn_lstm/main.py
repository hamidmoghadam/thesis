import lstm
import tensorflow as tf
import csv
import numpy as np
import data_provider as dp
import codecs

TRAINING_MESSAGE_ON = False

def main(train_user, train_data, validation_data, lst_test_data):
    word_2_id = dp.build_vocab(train_data)
    raw_train_data = dp.file_to_word_ids(train_data, word_2_id)
    raw_valid_data = dp.file_to_word_ids(validation_data, word_2_id)
    lst_raw_test_data = []
    for test_data in lst_test_data:
        lst_raw_test_data.append(dp.file_to_word_ids(test_data[1], word_2_id))

    config = lstm.BestConfig()
    eval_config = lstm.BestConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)

        with tf.name_scope("Train"):
            train_input = lstm.LSTMInput(
                config=config, data= raw_train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = lstm.LSTMNetwork(
                    is_training=True, config=config, input=train_input)
            #tf.summary.scalar("Training Loss", m.cost)
            #tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = lstm.LSTMInput(
                config=config, data=raw_valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = lstm.LSTMNetwork(
                    is_training=False, config=config, input=valid_input)

            #tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                lst_mtest = []
                for raw_test_data in lst_raw_test_data:
                    test_input = lstm.LSTMInput(
                        config=eval_config, data=raw_test_data, name="TestInput")
                    lst_mtest.append(lstm.LSTMNetwork(
                        is_training=False, config=config, input=test_input))

        sv = tf.train.Supervisor()
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i +
                                                  1 - config.max_epoch, 0.0)
                lr = config.learning_rate * lr_decay
                m.set_lr(lr, session)
                if TRAINING_MESSAGE_ON:
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, lr))

                train_perplexity = m.run_epoch(session)
                if TRAINING_MESSAGE_ON:
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

                valid_perplexity = mvalid.run_epoch(session)
                if TRAINING_MESSAGE_ON:
                    print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
            i = 0
            min_perplexity = 100
            min_perplexity_user = ''
            for mtest in lst_mtest:
                test_perplexity = mtest.run_epoch(session)
                if test_perplexity < min_perplexity:
                    min_perplexity = test_perplexity
                    min_perplexity_user = lst_test_data[i][0]
                print([lst_test_data[i][0], "Test Perplexity : %.3f" % test_perplexity])
                i += 1
            print('------------ {0} : {1} --------------'.format(train_user, min_perplexity_user))

user_count = 50
with open(r'../tumblr_twitter_scrapper/username_pair_filtered.csv', 'r', encoding='utf-8') as username_pair:
    reader = csv.reader(username_pair, delimiter=' ')
    lst_username_pair = []
    for row in reader:
        lst_username_pair.append((row[0], row[2].replace(r'twitter.com/', '')))

    for row in lst_username_pair[:user_count]:
        twitter_username = row[1]
        print(twitter_username)

        tweets_path = '../tumblr_twitter_scrapper/tweets/{0}.csv'.format(
            twitter_username)

        train_data = ''
        valid_data = ''

        with open(tweets_path, 'r', encoding='utf-8') as tweets_file:
            tweet_reader = csv.reader(tweets_file, delimiter=' ')
            lst_tweets = []
            for item in tweet_reader:
                if item[3] == 'True':
                    lst_tweets.append(item[1])
            if len(lst_tweets) < 10:
                continue
            np.random.shuffle(lst_tweets)
            trin_data_end_index = int(np.round(len(lst_tweets) * 0.7))

            train_data = '<eos>'.join(lst_tweets[:trin_data_end_index])
            valid_data = '<eos>'.join(lst_tweets[trin_data_end_index:])

        lst_test_data = []
        for item in lst_username_pair[:user_count]:
            tumblr_username = item[0]
            posts_path = '../tumblr_twitter_scrapper/posts/{0}.csv'.format(
                tumblr_username)
            test_data = ''

            with open(posts_path, 'r', encoding='utf-8') as tumblr_file:
                tumblr_reader = csv.reader(tumblr_file, delimiter=' ')
                lst_posts = []
                for post in tumblr_reader:
                    if post[5] == 'True':
                        lst_posts.append(post[4])
                if len(lst_posts) < 10:
                    continue
            test_data = '<eos>'.join(lst_posts)
            lst_test_data.append((tumblr_username, test_data))

        main(row[2], train_data, valid_data, lst_test_data)
