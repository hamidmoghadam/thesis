import csv
import numpy as np
from schema import *
import refine
import collections
import matplotlib.pyplot as plt


class data_provider(object):
    def __init__(self, size = 10, sent_max_len = 30, number_of_post_per_user = 50):
        self.train_batch_counter = 0
        self.valid_batch_counter = 0
        self.test_batch_counter = 0

        self.size = size

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


        user_selection_idx = np.random.permutation([x for x in range(31)])[:size]

        lst_tumblr_username = (np.array(lst_tumblr_username)[user_selection_idx]).tolist()
        lst_twitter_username = (np.array(lst_twitter_username)[user_selection_idx]).tolist()

        for twitter_username in lst_twitter_username:
            temp_set = []
            with open('../tumblr_twitter_scrapper/merged_tweets/{0}.csv'.format(twitter_username), 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    item = TwitterItem(row)
                    if item.is_owner == True:
                        content = refine.clean(item.content, ignore_url= True, ignore_stopword=False)
                        if len(content.split(' ')) > 2:
                            sents = refine.get_sentences(content)
                            content = ' <eos> '.join(sents)
                            temp_set.append(content)

            random_set = np.zeros(len(temp_set))
            train_count = int(np.round(len(temp_set) * 0.7))
            random_set[:min(train_count, number_of_post_per_user)] = 1
            np.random.shuffle(random_set)

            for i in range(len(temp_set)):
                if random_set[i] == 1:
                    train_data.append(temp_set[i])
                    k = lst_twitter_username.index(twitter_username)
                    label = [0 for x in range(size)]
                    label[k] = 1
                    y_train_data.append(label)
                else:
                    valid_data.append(temp_set[i])
                    k = lst_twitter_username.index(twitter_username)
                    label = [0 for x in range(size)]
                    label[k] = 1
                    y_valid_data.append(label)
        test_data = []
        y_test_data = []
        
        for tumblr_username in lst_tumblr_username:
            with open('../tumblr_twitter_scrapper/merged_posts/{0}.csv'.format(tumblr_username), 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=' ')
                temp_set = []
                y_temp_set = []
                for row in reader:
                    item = TumblrItem(row)
                    if item.is_owner:
                        content = refine.clean(item.content, ignore_url= True,ignore_stopword=False)
                        content_count = len(content.split(' '))
                        if(content_count < 3):
                            continue
                        if content_count > sent_max_len:
                            for sent in refine.get_sentences(content):
                                #test_data.append(sent)
                                temp_set.append(sent)
                                k = lst_tumblr_username.index(tumblr_username)
                                label = [0 for x in range(size)]
                                label[k] = 1
                                y_temp_set.append(label)
                                #y_test_data.append(label)
                        else: 
                            sents = []
                            for sent in refine.get_sentences(content):
                                sents.append(sent)
                            #test_data.append(' <eos> '.join(sents))
                            temp_set.append(' <eos> '.join(sents))
                            k = lst_tumblr_username.index(tumblr_username)
                            label = [0 for x in range(size)]
                            label[k] = 1
                            #y_test_data.append(label)
                            y_temp_set.append(label)
        
                random_set = np.zeros(len(test_data), dtype=bool)
                random_set[:number_of_post_per_user] = True
                np.random.shuffle(random_set)

                test_data.extend((np.array(temp_set)[random_set]).tolist())
                y_test_data.extend((np.array(y_temp_set)[random_set]).tolist())

        word_2_id = self.build_vocab(' '.join(train_data))
        self.vocab_size = len(word_2_id) + 1

        max_tweet_len = 0
        max_tweet = ''
        lst_len = []
        lst_len_test = []


        self.train_set = []
        self.valid_set = []
        self.test_set = []

        self.y_train_set = []
        self.y_valid_set = []
        self.y_test_set = []


        for i in range(len(train_data)):
            txt = train_data[i]
            lst_len.append(len(txt.split(' ')))
            temp = self.pad_word_ids(self.text_to_word_ids(txt, word_2_id), sent_max_len)
            if np.sum(temp) > 0:
                self.train_set.append(temp)
                self.y_train_set.append(y_train_data[i])


        for i in range(len(valid_data)):
            txt = valid_data[i]
            lst_len.append(len(txt.split(' ')))
            temp = self.pad_word_ids(self.text_to_word_ids(txt, word_2_id), sent_max_len)
            if np.sum(temp) > 0:
                self.valid_set.append(temp)
                self.y_valid_set.append(y_valid_data[i])


        for i in range(len(test_data)):
            txt = test_data[i]
            lst_len_test.append(len(txt.split(' ')))
            temp = self.pad_word_ids(self.text_to_word_ids(txt, word_2_id), sent_max_len)
            if np.sum(temp) > 0:
                self.test_set.append(temp)
                self.y_test_set.append(y_test_data[i])

        '''        
        t = sent_max_len

        plt.hist([x for x in lst_len if x <= t], facecolor='g')
        plt.figure()
        plt.hist([x for x in lst_len_test if x <= t], facecolor='b')
        plt.figure()
        plt.boxplot([[x for x in lst_len if x <= t], [x for x in lst_len_test if x <= t]])
        
        plt.show()
        '''
        
        self.train_size = len(self.train_set)
        self.valid_size = len(self.valid_set)
        self.test_size = len(self.test_set)

    def get_next_train_batch(self, batch_size):
        if(self.train_batch_counter >= self.train_size // batch_size):
            self.train_batch_counter = 0

        train = self.train_set[self.train_batch_counter * batch_size: (self.train_batch_counter+1) * batch_size]
        y_train = self.y_train_set[self.train_batch_counter * batch_size: (self.train_batch_counter+1) * batch_size]
       
        self.train_batch_counter += 1

        return (train, y_train)
    
    def get_next_valid_batch(self, batch_size):
        if(self.valid_batch_counter >= self.valid_size // batch_size):
            self.valid_batch_counter = 0
            
        valid = self.valid_set[self.valid_batch_counter * batch_size: (self.valid_batch_counter+1) * batch_size]
        y_valid = self.y_valid_set[self.valid_batch_counter * batch_size: (self.valid_batch_counter+1) * batch_size]
       
        self.valid_batch_counter += 1

        return (valid, y_valid)

    def get_next_test_batch(self):
        label = [0 for x in range(self.size)]
        label[self.test_batch_counter] = 1
        indices = [label == i for i in self.y_test_set]
        batch_size = len(indices)

        if(self.test_batch_counter == self.size):
            self.test_batch_counter = 0

        test = np.array(self.test_set)[indices]
        y_test = np.array(self.y_test_set)[indices]
       
        self.test_batch_counter += 1

        return (test.tolist(), y_test.tolist())

    def get_next_test_batch(self, class_id):
        label = [0 for x in range(self.size)]
        label[class_id] = 1
        indices = [label == i for i in self.y_test_set]
        batch_size = len(indices)

        test = np.array(self.test_set)[indices]
        y_test = np.array(self.y_test_set)[indices]

        return (test.tolist(), y_test.tolist())


    def build_vocab(self,data):
        #data = _read_words(filename)

        counter = collections.Counter(data.split(' '))
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(1, len(words)+1)))
        #print("vocab size is {0}".format(len(word_to_id)))
        return word_to_id


    def text_to_word_ids(self,data, word_to_id):
        #data = _read_words(filename)
        return [word_to_id[word] for word in data.split(' ') if word in word_to_id]

    def pad_word_ids(self, word_ids, max_length):
        data_len = len(word_ids)
            
        if data_len < max_length:
            word_ids = np.lib.pad(word_ids, (max_length - data_len, 0), 'constant').tolist()

        return word_ids[:max_length]
