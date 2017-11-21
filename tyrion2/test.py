import csv
import numpy as np
from schema import *
import refine


print(np.random.randint(10,size=1)[0])

'''

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


for twitter_username in lst_twitter_username[:1]:
    temp_set = []
    with open('../tumblr_twitter_scrapper/merged_tweets/{0}.csv'.format(twitter_username), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ')
        i = 0
        for row in reader:
            item = TwitterItem(row)
            if item.is_owner == True:
                content = refine.clean(item.content, ignore_url= True, ignore_stopword=False, ignore_digit=False)
                print(content)
                sents = refine.get_sentences(content)
                print(' <eos> '.join(sents))
                i += 1
                if i == 10:
                    break

        '''