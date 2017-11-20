from selenium import webdriver
from bs4 import BeautifulSoup
import time
import os
import csv
import re
import random

CHROME_DRIVER_PATH = '/Documents/codes/tumbler_scrapper/chromedriver'

def crawl_users_follower(username, html):
    follower_set = []

    with open('files/tumblr_{0}_follower.csv'.format(username), 'a+') as follower_file:
        reader = csv.reader(follower_file, delimiter= ' ')
        for row in reader:
            follower_file.insert(row[0])

    notes = html.find_all('ol', {'class': 'notes'})
    if len(notes) > 0:
        notes = str(notes[0])
        likes = re.findall(r'https://[a-zA-Z0-9_-]+\.tumblr\.com/', notes)
        likes = list(set(likes))
        with open('files/tumblr_{0}_follower.csv'.format(username), 'a') as follower_file:
            follower_writer = csv.writer(follower_file, delimiter=' ')
            for like in likes:
                if like not in follower_set:
                    follower_writer.writerow([like])


def crawl_users_pages():
    fetched_user_set = set()
    with open('user_data_fetched.csv', 'r') as user_data_fetched:
        reader = csv.reader(user_data_fetched, delimiter = ' ')
        for row in reader:
            fetched_user_set.add(row[0])
    print(fetched_user_set)
    with open('user_data_fetched.csv', 'a') as user_data_fetched:
        write_fetched_user = csv.writer(user_data_fetched, delimiter = ' ')
        with open('username_pairs.csv', 'r') as username_pair_file:
            reader = csv.reader(username_pair_file, delimiter=' ')
            browser = webdriver.Chrome()
            posts = set()
            for row in reader:
                if row[0] in fetched_user_set:
                    continue
                try:
                    for i in range(1, 1):
                        time.sleep(10)
                        browser.get(row[1] + '/page/' + str(i))
                        page_source = browser.page_source
                        temp = re.findall(row[1] + r'/post/[0-9]+/[a-zA-Z0-9_@-]+', page_source)
                        if len(temp) == 0:
                            break
                        for item in temp:
                            posts.add(item)
                        
                        with open('files/tumblr_{0}_page.csv'.format(row[0]), 'a+') as content_file:
                            writer = csv.writer(content_file, delimiter=' ')
                            src = browser.page_source
                            html = BeautifulSoup(src, 'lxml')
                            for p in html.find_all('p'):
                                writer.writerow([p.getText()])
                    for post in posts:
                        try:
                            time.sleep(10)
                            browser.get(post)
                            src = browser.page_source
                            if src.find('more_notes_link'):
                                updated_src = ''
                                while src != updated_src and src.find('more_notes_link'):
                                    updated_src = src
                                    browser.execute_script("document.getElementsByClassName('more_notes_link').click();")
                                    src = browser.page_source

                            
                            html = BeautifulSoup(src, 'lxml')
                            crawl_users_follower(row[0], html)
                        except:
                            print('fail to crawl a post')
                    write_fetched_user.writerow([row[0]])
                except:
                       print('fail to crawl {0}'.format(row[0]))     
            browser.close()
        
crawl_users_pages()