import csv
import numpy as np
from schema import *
import refine
import collections
import enchant


ench = enchant.Dict('en_US')

lst_username = []
lst_twitter_username = []
lst_tumblr_username = []
        
with open('../tumblr_twitter_scrapper/large_username_pairs_filtered.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        if int(row[4]) >= 2000 :
            lst_username.append({'tumblr': row[0], 'twitter': row[2].replace(r'twitter.com/', '')})
            lst_twitter_username.append(row[2].replace(r'twitter.com/', ''))
            lst_tumblr_username.append(row[0])

            

number_of_words = 0
number_of_miss = 0
dic_miss = dict()

print(len(lst_twitter_username))
print(lst_twitter_username[0])
for twitter_username in lst_twitter_username[:1]:
    
    with open('../tumblr_twitter_scrapper/merged_tweets/{0}.csv'.format(twitter_username), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            item = TwitterItem(row)
            if item.is_owner == True:
                content = refine.clean(item.content, ignore_url= False, ignore_stopword=False)
                for word in content.split(' '):
                    if not ench.check(word):
                        number_of_miss += 1
                        if word not in dic_miss:
                            dic_miss[word] = 1
                        else:
                            dic_miss[word] += 1
                    number_of_words +=1

print(dic_miss)
print(number_of_miss)
print(number_of_words)
                