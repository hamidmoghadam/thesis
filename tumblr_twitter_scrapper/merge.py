import csv
import numpy as np

def merge_twitter():
    lst_user_pair = []
    with open('large_username_pairs_filtered.csv', 'r') as username_pairs:
        reader = csv.reader(username_pairs, delimiter= ' ')
        for row in reader:
            lst_user_pair.append([row[0], row[2].replace('twitter.com/', '')])

    for pair in lst_user_pair:
        twitter_username = pair[1]
        print(twitter_username)
        lst_tweet = []
        with open('merged_tweets/{0}.csv'.format(twitter_username), 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                lst_tweet.append(row[1])
        repeated_tweet = 0
        added_tweet = 0
        with open ('merged_tweets/{0}.csv'.format(twitter_username), 'a') as w:
            writer = csv.writer(w, delimiter= ' ')
            with open('tweets2/{0}.csv'.format(twitter_username), 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if row[1] not in lst_tweet:
                        writer.writerow(row)
                        added_tweet += 1
                    else:
                        repeated_tweet +=1
        print('repeated tweet : {0} and added tweet : {1}'.format(repeated_tweet, added_tweet))
                    
def merge_tumblr():
    lst_user_pair = []
    with open('large_username_pairs_filtered.csv', 'r') as username_pairs:
        reader = csv.reader(username_pairs, delimiter= ' ')
        for row in reader:
            lst_user_pair.append([row[0], row[2].replace('twitter.com/', '')])

    for pair in lst_user_pair:
        tumblr_username = pair[0]
        print(tumblr_username)
        lst_post = []
        with open('merged_posts/{0}.csv'.format(tumblr_username), 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                lst_post.append(row[4])
        repeated_post = 0
        added_post = 0
        with open ('merged_posts/{0}.csv'.format(tumblr_username), 'a') as w:
            writer = csv.writer(w, delimiter= ' ')
            with open('posts2/{0}.csv'.format(tumblr_username), 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if row[4] not in lst_post:
                        writer.writerow(row)
                        added_post += 1
                    else:
                        repeated_post +=1
        print('repeated post : {0} and added post : {1}'.format(repeated_post, added_post))
    
merge_tumblr()
