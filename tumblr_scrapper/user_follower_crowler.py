from selenium import webdriver
from bs4 import BeautifulSoup
import time
import os
import csv
import re
import random
from pathlib import Path

CHROME_DRIVER_PATH = '/Users/hamidmoghaddam/Documents/codes/tumblr_scrapper/chromedriver'


def extract_and_save_follower(username, src):
    follower_set = set()
    html = BeautifulSoup(src, 'lxml')
    file = Path('followers/{0}.csv'.format(username))
    if file.exists():
        with open('followers/{0}.csv'.format(username), 'r') as follower_file:
            reader = csv.reader(follower_file, delimiter=' ')
            for row in reader:
                follower_set.add(row[0])

    notes = html.find_all('ol', {'class': 'notes'})
    if len(notes) > 0:
        notes = str(notes[0])
        likes = re.findall(r'https://[a-zA-Z0-9_-]+\.tumblr\.com/', notes)
        likes = list(set(likes))
        with open('followers/{0}.csv'.format(username), 'a') as follower_file:
            follower_writer = csv.writer(follower_file, delimiter=' ')
            for like in likes:
                if like not in follower_set:
                    follower_writer.writerow([like])


fetched_user_set = set()
with open('user_followers_fetched.csv', 'r') as user_followers_fetched:
    reader = csv.reader(user_followers_fetched, delimiter=' ')
    for row in reader:
        fetched_user_set.add(row[0])

lst_username = []

with open('username_pairs.csv', 'r') as username_pair:
    reader = csv.reader(username_pair, delimiter=' ')
    for row in reader:
        lst_username.append(row[0])

browser = webdriver.Chrome()

with open('user_followers_fetched.csv', 'a') as user_followers_fetched:
    write_fetched_followers = csv.writer(user_followers_fetched, delimiter=' ')
    for username in lst_username:
        if username in fetched_user_set:
            continue
        print('{0} follower crawling starts'.format(username))
        with open('posts/{0}.csv'.format(username)) as user_posts:
            reader = csv.reader(user_posts, delimiter=' ')
            for row in reader:
                if row[4] == False:
                    continue
                time.sleep(5)
                post_url = row[2]
                try:
                    browser.get(post_url)
                    src = browser.page_source
                    try:
                        if src.find('more_notes_link') > 0:
                            updated_src = ''
                            while src != updated_src and src.find('more_notes_link') > 0:
                                link = browser.find_element_by_css_selector(".more_notes_link")
                                if not link.is_displayed():
                                    break
                                updated_src = src
                                link.click()
                                time.sleep(4)
                                src = browser.page_source
                    except:
                        pass

                    extract_and_save_follower(username, src)
                except Exception as e:
                    print(e)

        write_fetched_followers.writerow([username])
        user_followers_fetched.flush()
        print('{0} follower crawling finished'.format(username))
