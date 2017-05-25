import pytumblr
import csv
from bs4 import BeautifulSoup
import time
import tweepy
import json
import os
import sys
from time import sleep
from tweepy.error import TweepError, RateLimitError, is_rate_limit_error_message
import csv
import pandas as pd



def retrieve_tumblr_posts(username):
    # Authenticate via OAuth
    client = pytumblr.TumblrRestClient(
        'cUaEmGIleDG46g7L58aeNhtcwMZlWRLXTPRKYsEPlzybMzHtxV',
        'Ra1jDVcpjLQD81tQwSl4Lk9iSABDDQ7DNfPgedGaMoGI4VP5WW',
        't81wrl8ZQPKnEdhOANppgwl4KkXgoadZtfby3z044BxfTcGhmw',
        'n8vEi9Y5sHRkfNPOfyDnR7lBeeMF2z2S2aV7OmBd2w5HaVBGw9'
    )

    succeed = False

    print('tumblr: {0} crawling started'.format(username))
    with open('posts/{0}.csv'.format(username), 'w') as post_file:
        writer = csv.writer(post_file, delimiter=' ')
        try:
            data = client.posts(username, 'text')
            for post in data['posts']:
                id = post['id']
                trail = post['trail']
                user_is_owner = True
                if len(trail) > 0:
                    user_is_owner = trail[0]['post']['id'] == str(id)
                date = post['date']
                body = post['body'].replace('\n', ' ')
                body = BeautifulSoup(body, 'lxml').get_text()
                url = post['post_url']

                writer.writerow([username, id, date, url, body, user_is_owner, 'text'])

            data = client.posts(username, 'link')
            for post in data['posts']:
                id = post['id']
                trail = post['trail']
                user_is_owner = True
                if len(trail) > 0:
                    user_is_owner = trail[0]['post']['id'] == str(id)
                date = post['date']
                body = post['description'].replace('\n', ' ')
                body = BeautifulSoup(body, 'lxml').get_text()
                url = post['post_url']

                writer.writerow([username, id, date, url, body, user_is_owner, 'link'])
            print('tumblr: {0} crawling finished'.format(username))
            succeed = True
        except Exception as e:
            print('Error in fetching {0} posts'.format(username))
            print(e)
            succeed = False

    return succeed

def retrieve_twitter_tweets(username):
    CONSUMER_KEY = 'zBVADfE51fuljBWABOef1RwXd'
    CONSUMER_SECRET = '5oieqtlgBjuzqo93sU6g4HJo5ISTpJzxb6Ek19TGPHHPOe9SJN'
    ACCESS_TOKEN = '757496781630533632-n6lhsqDif1IcgDkEcFd5UjWciI22Ncg'
    ACCESS_TOKEN_SECRET = 'M7WcXHcVDhdpEGPpgo1TvsVDbShztzOLtrQa1sxZZ9wdi'

    def limit_handled(cursor):
        while True:
            try:
                yield cursor.next()
            except Exception as e:
                if '429' in str(e) or '88' in str(e):
                    print("... sleeping ....\n")
                    sleep(15 * 60)
                    print("... woken up ...")
                else:
                    raise e

    # Create an Api instance.
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    api = tweepy.API(auth)

    succeed = False
    print('twitter: {0} crawling started'.format(username))
    with open('tweets/{0}.csv'.format(username), 'w') as tweet_file:
        writer = csv.writer(tweet_file, delimiter=' ')
        try:
            for t in limit_handled(tweepy.Cursor(api.user_timeline, screen_name=username).items()):
                timeline = t._json
                text = timeline['text']
                created_at = timeline['created_at']
                writer.writerow([username, text, created_at, not text.startswith('RT')])
            succeed = True
            print('twitter: {0} crawling finished'.format(username))
        except Exception as e:
            print('Error in fetching {0} tweets'.format(username))
            print(e)
            succeed = False

    return succeed


fetched_user_set = set()
with open('user_data_fetched.csv', 'r') as user_data_fetched:
    reader = csv.reader(user_data_fetched, delimiter=' ')
    for row in reader:
        fetched_user_set.add(row[0])

with open('username_pair_filtered.csv', 'r') as username_pairs:
    reader = csv.reader(username_pairs, delimiter= ' ')
    with open('user_data_fetched.csv', 'a') as user_data_fetched:
        write_fetched_user = csv.writer(user_data_fetched, delimiter=' ')
        for row in reader:
            tumblr_username = row[0]
            twitter_username = row[2].replace('twitter.com/', '');

            if tumblr_username in fetched_user_set:
                continue

            retrieve_tumblr_posts(tumblr_username)
            retrieve_twitter_tweets(twitter_username)

            fetched_user_set.add(tumblr_username)
            write_fetched_user.writerow([tumblr_username])
            user_data_fetched.flush()

            time.sleep(10)
        #pdb.set_trace()



#client.avatar('codingjester') # get the avatar for a blog
#print(client.blog_likes('reincepriebus')) # get the likes on a blog
#print(client.followers('reincepriebus')) # get the followers of a blog
#client.queue('codingjester') # get the queue for a given blog
#client.submission('codingjester') # get the submissions for a given blog
