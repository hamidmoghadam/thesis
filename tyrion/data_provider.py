import csv
import numpy as np
from schema import *
import refine
import collections

class data_provider(object):
    def __init__(self, size = 10, sent_max_len = 30):
        self.train_batch_counter = 0
        self.valid_batch_counter = 0
        self.test_batch_counter = 0

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
        self.y_train_data = []
        valid_data = []
        self.y_valid_data = []


        for twitter_username in lst_twitter_username[:size]:
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
                        temp_set.append(content)

                random_set = np.zeros(len(temp_set))
                train_count = int(np.round(len(temp_set) * 0.7))
                random_set[:train_count] = 1
                np.random.shuffle(random_set)

                for i in range(len(temp_set)):
                    if random_set[i] == 1:
                        train_data.append(temp_set[i])
                        k = lst_twitter_username.index(twitter_username)
                        label = [0 for x in range(size)]
                        label[k] = 1
                        self.y_train_data.append(label)
                    else:
                        valid_data.append(temp_set[i])
                        k = lst_twitter_username.index(twitter_username)
                        label = [0 for x in range(size)]
                        label[k] = 1
                        self.y_valid_data.append(label)
        test_data = []
        self.y_test_data = []

        for tumblr_username in lst_tumblr_username[:size]:
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
                                label = [0 for x in range(size)]
                                label[k] = 1
                                self.y_test_data.append(label)

        word_2_id = self.build_vocab(' '.join(train_data))

        max_tweet_len = 0
        max_tweet = ''
        lst_len = []


        self.train_set = []
        self.valid_set = []
        self.test_set = []


        for txt in train_data:
            temp = self.pad_word_ids(self.text_to_word_ids(txt, word_2_id), sent_max_len)
            if np.sum(temp) > 0:
                self.train_set.append(temp)


        for txt in valid_data:
            temp = self.pad_word_ids(self.text_to_word_ids(txt, word_2_id), sent_max_len)
            if np.sum(temp) > 0:
                self.valid_set.append(temp)


        for txt in test_data:
            temp = self.pad_word_ids(self.text_to_word_ids(txt, word_2_id), sent_max_len)
            if np.sum(temp) > 0:
                self.test_set.append(temp)

        self.train_size = len(self.train_set)
        self.valid_size = len(self.valid_set)
        self.test_size = len(self.test_set)

    def get_next_train_batch(self, batch_size):
        if(self.train_batch_counter > self.train_size // batch_size):
            self.train_batch_counter = 0

        train = self.train_set[self.train_batch_counter * batch_size: (self.train_batch_counter+1) * batch_size]
        y_train = self.y_train_data[self.train_batch_counter * batch_size: (self.train_batch_counter+1) * batch_size]
       
        self.train_batch_counter += 1

        return (train, y_train)
    
    def get_next_valid_batch(self, batch_size):
        if(self.valid_batch_counter > self.valid_size // batch_size):
            self.valid_batch_counter = 0
            
        valid = self.valid_set[self.valid_batch_counter * batch_size: (self.valid_batch_counter+1) * batch_size]
        y_valid = self.y_valid_data[self.valid_batch_counter * batch_size: (self.valid_batch_counter+1) * batch_size]
       
        self.valid_batch_counter += 1

        return (valid, y_valid)

    def get_next_test_batch(self, batch_size):
        if(self.test_batch_counter > self.test_size // batch_size):
            self.test_batch_counter = 0
            
        test = self.test_set[self.test_batch_counter * batch_size: (self.test_batch_counter+1) * batch_size]
        y_test = self.y_test_data[self.test_batch_counter * batch_size: (self.test_batch_counter+1) * batch_size]
       
        self.test_batch_counter += 1

        return (test, y_test)

    def build_vocab(self,data):
        #data = _read_words(filename)

        counter = collections.Counter(data.split(' '))
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(1, len(words)+1)))
        print("vocab size is {0}".format(len(word_to_id)))
        return word_to_id


    def text_to_word_ids(self,data, word_to_id):
        #data = _read_words(filename)
        return [word_to_id[word] for word in data.split(' ') if word in word_to_id]

    def pad_word_ids(self, word_ids, max_length):
        data_len = len(word_ids)
            
        if data_len < max_length:
            word_ids = np.lib.pad(word_ids, (max_length - data_len, 0), 'constant').tolist()

        return word_ids[:max_length]
