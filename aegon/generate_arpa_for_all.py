import printer
from subprocess import call
from schema import *
import csv

with open(r'../tumblr_twitter_scrapper/large_username_pairs_filtered.csv', 'r', encoding='utf-8') as username_pair:
    reader = csv.reader(username_pair, delimiter=' ')
    lst_username_pair = []

    for row in reader:
        tumblr_username = row[0]
        twitter_username = row[2].replace(r'twitter.com/', '')
        with open("../tumblr_twitter_scrapper/tweets/text/{0}.txt".format(twitter_username), 'w') as f:
            for txt in printer.provide('twitter', twitter_username):
                f.write(txt)
        with open("../tumblr_twitter_scrapper/posts/text/{0}.txt".format(tumblr_username), 'w') as f:
            for txt in printer.provide('tumblr', tumblr_username):
                f.write(txt)
        with open("build_language_model.sh", 'a+') as f:
            f.write('../kenlm/bin/lmplz -o 3 <../tumblr_twitter_scrapper/tweets/text/{0}.txt > ../tumblr_twitter_scrapper/tweets/arpa/{0}.arpa\n'.format(twitter_username))       
            f.write('../kenlm/bin/lmplz -o 3 <../tumblr_twitter_scrapper/posts/text/{0}.txt > ../tumblr_twitter_scrapper/posts/arpa/{0}.arpa\n'.format(tumblr_username))
        



