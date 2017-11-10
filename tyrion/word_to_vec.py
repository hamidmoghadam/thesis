import gensim
import csv
import numpy as np
from schema import *
import refine
import collections
import matplotlib.pyplot as plt



lst_username = []
lst_twitter_username = []
lst_tumblr_username = []
with open('../tumblr_twitter_scrapper/large_username_pairs_filtered.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        lst_username.append({'tumblr': row[0], 'twitter': row[2].replace(r'twitter.com/', '')})
        lst_twitter_username.append(row[2].replace(r'twitter.com/', ''))
        lst_tumblr_username.append(row[0])

temp_set = []
for twitter_username in lst_twitter_username:
    with open('../tumblr_twitter_scrapper/merged_tweets/{0}.csv'.format(twitter_username), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            item = TwitterItem(row)
            if item.is_owner == True:
                content = refine.clean(item.content, ignore_url= True, ignore_stopword=False)
                if len(content.split(' ')) > 2:
                    sents = refine.get_sentences(content)
                    for sent in sents:
                        temp_set.append(sent.split(' '))

model = gensim.models.Word2Vec(temp_set, size=70, window=1, min_count=0)
model.save('preTrainedEmbedding.txt')