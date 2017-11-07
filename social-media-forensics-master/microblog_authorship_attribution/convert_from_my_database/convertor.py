import csv
import numpy as np
from schema import *
import refine
import collections
import matplotlib.pyplot as plt
from time import gmtime, strftime
import sys
from string import Template

lst_username = []
lst_twitter_username = []
lst_tumblr_username = []

f = open('../../../tumblr_twitter_scrapper/large_username_pairs_filtered.csv', 'r')
reader = csv.reader(f, delimiter=' ')
for row in reader:
    lst_username.append(
        {'tumblr': row[0], 'twitter': row[2].replace(r'twitter.com/', '')})
    lst_twitter_username.append(row[2].replace(r'twitter.com/', ''))
    lst_tumblr_username.append(row[0])
f.close()

tweet_code = 1
s = Template('$username $time $code{\n$tweet\n#POS #POS\n}\n')
for username in lst_username:
    tumblr_username = username['tumblr']
    twitter_username = username['twitter']
    f_twitter = open('../../../tumblr_twitter_scrapper/merged_tweets/{0}.csv'.format(twitter_username), 'r')
    f_tumblr = open('../../../tumblr_twitter_scrapper/merged_posts/{0}.csv'.format(tumblr_username), 'r')
    twitter_set = []
    tumblr_set = []

    reader = csv.reader(f_twitter, delimiter=' ')
    for row in reader:
        item = TwitterItem(row)
        if item.is_owner == True:
            content = refine.clean(unicode(item.content, 'utf-8'), ignore_url= True, ignore_stopword=False)
            if len(content.split(' ')) > 2:
                twitter_set.append(content)
    f_twitter.close()

    reader = csv.reader(f_tumblr, delimiter=' ')
    for row in reader:
        item = TumblrItem(row)
        if item.is_owner:
            content = refine.clean(unicode(item.content, 'utf-8'), ignore_url= True,ignore_stopword=False)
            content_count = len(content.split(' '))
            if(content_count > 3):
                if content_count > 100:
                    for sent in refine.get_sentences(content):
                        tumblr_set.append(sent)#refine.stem(sent))
                else: 
                    sents = []
                    for sent in refine.get_sentences(content):
                        sents.append(sent)#refine.stem(sent))
                    tumblr_set.append(' '.join(sents))
                        
    f_tumblr.close()

    w_twitter = open('train/{0}_{1}.dat'.format(len(twitter_set), twitter_username), 'w+')
    w_tumblr = open('test/{0}_{1}.dat'.format(len(twitter_set), twitter_username), 'w+')

    for row in twitter_set:
        tweet_code = tweet_code + 1
        body = s.substitute(username = twitter_username, time= strftime("%Y-%m-%d %H:%M:%S", gmtime()), code=tweet_code, tweet=row)
        w_twitter.write(body.encode("UTF-8"))
    w_twitter.close()

    for row in tumblr_set:
        tweet_code = tweet_code + 1
        body = s.substitute(username = twitter_username, time= strftime("%Y-%m-%d %H:%M:%S", gmtime()), code=tweet_code, tweet=row)
        w_tumblr.write(body.encode("UTF-8"))
    w_tumblr.close()