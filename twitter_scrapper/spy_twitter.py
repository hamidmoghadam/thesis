
from __future__ import print_function
import tweepy
import json
import os
import sys
from time import sleep
from tweepy.error import TweepError, RateLimitError, is_rate_limit_error_message
import csv
import pandas as pd

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
data = pd.read_csv('username_pairs.csv', delimiter=' ')


for u in data.values:
	try:
		username = u[2].replace('twitter.com/', '');
		target_file = open('files/twitter_data_{0}.json'.format(username), 'a')
		user = {}
		
		user['twitter_username'] = u[2];
		user['tumblr_username'] = u[1];


		print(username)
		#request get time line
		
		timeline = []
		for t in limit_handled(tweepy.Cursor(api.user_timeline, screen_name=username).items()):
			timeline.append(t._json)
			
			
		user['timeline'] = timeline
		
		#request get friends
		
		friends = []
		for t in limit_handled(tweepy.Cursor(api.friends_ids, screen_name=username).items()):
			friends.append(t)
			
		user['friends'] = friends
		
		#request get followers
		followers = []
		for t in limit_handled(tweepy.Cursor(api.followers_ids, screen_name=username).items()):
			followers.append(t)
			
		user['followers'] = followers
		
		
		json.dump(user, target_file)
		target_file.write(os.linesep)
		target_file.write('**** next user ****\n')
		target_file.close()
	except Exception as e: 
		log_file = open('error.log', 'a')
		log_file.write("Failed to crawl user {0}: {1}\n".format(u[2], str(e)));
		log_file.close()		
		print(e);
						
		




