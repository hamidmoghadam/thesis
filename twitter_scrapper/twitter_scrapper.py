
from selenium import webdriver
from bs4 import BeautifulSoup
import time
import os
import csv
import re
import random

CHROME_DRIVER_PATH = '/Documents/codes/tumbler_scrapper/chromedriver'
CHROME_DRIVER_PATH = '/Users/hamidmoghaddam/Documents/codes/tumblr_scrapper/chromedriver'


twitter_urls = []
with open('username_pair_filtered.csv', 'r') as username_pair_file:
    reader = csv.reader(username_pair_file, delimiter= ' ')
    for row in reader:
        twitter_urls.append(row[2])

browser = webdriver.Chrome(CHROME_DRIVER_PATH)

for url in twitter_urls:
    username = url.replace(r'twitter.com/', '')
    print(username)
    is_retweet = False
    lst_tweet_content = []
    browser.get('https://' + url)

    src_updated = browser.page_source
    src = ""
    max_scroll_count = 100
    i = 0
    while src != src_updated and i < max_scroll_count:
        time.sleep(40)
        src = src_updated
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        src_updated = browser.page_source
        i += 1

    html = BeautifulSoup(src, 'lxml')
    tweets = html.find_all('div', {'data-tweet-id' : re.compile('.*')})
    for div_tweet in tweets:
        div_retweet = div_tweet.find_all('span', {'class': 'js-retweet-text'})
        if len(div_retweet) > 0:
            is_retweet = True
        div_content = div_tweet.find_all('div', {'class': 'js-tweet-text-container'})
        if len(div_content) > 0:
            p = div_content[0].find('p')
            lst_tweet_content.append([p.getText(), is_retweet])
    with open('tweets/{0}.csv'.format(username), 'w') as f:
        writer = csv.writer(f, delimiter = ' ')
        for item in lst_tweet_content:
            writer.writerow([username, item[0], item[1]])

browser.close()
