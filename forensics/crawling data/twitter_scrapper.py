from selenium import webdriver
from bs4 import BeautifulSoup
import time
import os
import csv
import re
import random

#CHROME_DRIVER_PATH = '/Documents/codes/tumbler_scrapper/chromedriver'
#CHROME_DRIVER_PATH = '/Documents/codes/tumbler_scrapper/chromedriver'
CHROME_DRIVER_PATH = '/Users/hamidmoghaddam/Documents/codes/tumblr_scrapper/chromedriver'


def crawl_username():
    user_set = set()
    with open('datasets/username_twitter.csv', 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            user_set.add(row[0])

    
    urls = set()
    queries = 'a about above after all although\
                am among an and another any\
                anybody anyone anything are around as\
                at be because before behind below\
                beside between both but by can\
                cos do down each either enough\
                every everybody everyone everything few following\
                for from have he her him\
                i if in including inside into\
                is it its latter less like\
                little lots many me more most\
                much must my near need neither\
                no nobody none nor nothing of\
                off on once one onto opposite\
                or our outside over own past\
                per plenty plus regarding same several\
                she should since so some somebody\
                someone something such than that the\
                their them these they this those\
                though through till to toward towards\
                under unless unlike until up upon\
                us used via we what whatever\
                when where whether which while who\
                whoever whom whose will with within\
                without worth would yes you'

    browser = webdriver.Chrome(CHROME_DRIVER_PATH)

    for q in queries.split():
        print(q)
        url = 'https://twitter.com/search?f=users&vertical=default&q=' + q
        browser.get(url)
        src_updated = browser.page_source
        src = ""
        try:
            while src != src_updated:
                time.sleep(5)
                src = src_updated
                browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                src_updated = browser.page_source
        except Exception as e:
            print(e)

        html = BeautifulSoup(src_updated, 'lxml')
        spans = html.find_all('span', {'class': 'username u-dir'})

        for span in spans:
            if span.text not in user_set:
                with open('datasets/username_twitter.csv', 'a') as username_file:
                    writer = csv.writer(username_file, delimiter=' ')
                    username = span.get_text()
                    writer.writerow([username])
                    user_set.add(username)

    browser.close()
        

def make_unique_username():
    with open('username_tumblr.csv', 'r') as username_file:
        with open('username_tumblr_unique.csv', 'w') as username_file_unique:
            reader = csv.reader(username_file, delimiter=' ')
            writer = csv.writer(username_file_unique, delimiter=' ')
            user_set = set()
            for row in reader:
                if(row[0] not in user_set):
                    writer.writerow([row[0], row[1]])
                    user_set.add(row[0])
                    

def long_substr(data):
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and is_substr(data[0][i:i+j], data):
                    substr = data[0][i:i+j]
    return substr

def is_substr(find, data):
    if len(data) < 1 and len(find) < 1:
        return False
    for i in range(len(data)):
        if find not in data[i]:
            return False
    return True

def crawl_twitter_users():
    username_pairs = set()
    with open('username_pairs.csv', 'r') as username_pairs_file:
        reader = csv.reader(username_pairs_file, delimiter = ' ')
        for row in reader:
            username_pairs.add(row[0])
    username_blocked = set()
    with open('username_blocked.csv', 'r') as username_blocked_file:
        reader = csv.reader(username_blocked_file, delimiter = ' ')
        for row in reader:
            username_blocked.add(row[0])
    
    print(username_blocked)
            
    with open('username_tumblr.csv', 'r') as username_file:
        reader = csv.reader(username_file, delimiter=' ')
        with open('username_blocked.csv', 'a+') as username_blocked_file:
            writer_blocked = csv.writer(username_blocked_file, delimiter=' ')
            with open('username_pairs.csv', 'a+') as username_pairs_file:
                writer = csv.writer(username_pairs_file, delimiter=' ')
                browser = webdriver.Chrome()
                link_set = set()
                for row in reader:
                    twitter_url_set = set()
                    if row[0] in username_pairs or row[0] in username_blocked:
                        continue

                    url = row[1]
                    time.sleep(3)
                    browser.get(url)

                    src = browser.page_source
                    if len(src) == 0:
                        continue
                    src = src.replace('\'', '"')
                    src = re.sub('<p>.*<\/p>', '', src)
                    #twitter_links = re.findall(r'(twitter\.com/[^ ./\+\?]+)"', src)
                    twitter_links = re.findall(r'(twitter\.com/[a-zA-Z0-9_@]+)"', src)

                    for link in twitter_links:
                        if link not in link_set:
                            link_set.add(link)
                            twitter_url_set.add(link)

                    if len(twitter_url_set) > 0 :
                        selected_url = list(twitter_url_set)[0] 
                        common_text_len = 0
                        for twitter_url in twitter_url_set:
                            twitter_id = twitter_url[12:]
                            common_text = long_substr([row[0], twitter_id])
                            if len(common_text) > common_text_len :
                                common_text_len = len(common_text)
                                selected_url = twitter_url

                        writer.writerow([row[0], url, selected_url, common_text_len])
                        username_pairs.add(row[0])
                        print([row[0], selected_url])
                        username_pairs_file.flush()
                    else:
                        username_blocked.add(row[0])
                        writer_blocked.writerow([row[0]])
                        print([row[0], 'blocked!'])
                        username_blocked_file.flush()


def crawl_twitter_users_from_follower():
    username_pairs = set()
    with open('username_pairs.csv', 'r') as username_pairs_file:
        reader = csv.reader(username_pairs_file, delimiter=' ')
        for row in reader:
            username_pairs.add(row[0])
    username_blocked = set()
    with open('username_follower_blocked.csv', 'r') as username_blocked_file:
        reader = csv.reader(username_blocked_file, delimiter=' ')
        for row in reader:
            username_blocked.add(row[0])

    print(username_blocked)

    with open('all_follower.csv', 'r') as username_file:
        reader = csv.reader(username_file, delimiter=' ')
        with open('username_follower_blocked.csv', 'a+') as username_blocked_file:
            writer_blocked = csv.writer(username_blocked_file, delimiter=' ')
            with open('username_pairs.csv', 'a+') as username_pairs_file:
                writer = csv.writer(username_pairs_file, delimiter=' ')
                browser = webdriver.Chrome()
                link_set = set()
                for row in reader:
                    try:
                        twitter_url_set = set()
                        username = row[0].replace(r'https://', '').replace('.tumblr.com/', '')
                        if username in username_pairs or username in username_blocked:
                            print('.')
                            continue

                        url = row[0]
                        #time.sleep(3)
                        browser.get(url)

                        src = browser.page_source
                        if len(src) == 0:
                            continue
                        src = src.replace('\'', '"')
                        src = re.sub('<p>.*<\/p>', '', src)
                        # twitter_links = re.findall(r'(twitter\.com/[^ ./\+\?]+)"', src)
                        twitter_links = re.findall(r'(twitter\.com/[a-zA-Z0-9_@]+)"', src)

                        for link in twitter_links:
                            if link not in link_set:
                                link_set.add(link)
                                twitter_url_set.add(link)

                        if len(twitter_url_set) > 0:
                            selected_url = list(twitter_url_set)[0]
                            common_text_len = 0
                            for twitter_url in twitter_url_set:
                                twitter_id = twitter_url[12:]
                                common_text = long_substr([username, twitter_id])
                                if len(common_text) > common_text_len:
                                    common_text_len = len(common_text)
                                    selected_url = twitter_url

                            writer.writerow([username, url, selected_url, common_text_len])
                            username_pairs.add(username)
                            print([username, selected_url])
                            username_pairs_file.flush()
                        else:
                            username_blocked.add(username)
                            writer_blocked.writerow([username])
                            print([username, 'blocked!'])
                            username_blocked_file.flush()
                    except:
                        #username_blocked.add(username)
                        #writer_blocked.writerow([username])
                        print([username, 'Passed with error!'])





def crawl_users_pages():
    with open('username_pairs.csv', 'r') as username_pair_file:
        reader = csv.reader(username_pair_file, delimiter= ' ')
        browser = webdriver.Chrome()
        posts = []
        for row in reader:
            for i in range(1,20):
                page_source = browser.page_source
                posts.extend(re.findall(r'http://'+row[1]+'/post/[0-9]+/[a-zA-Z0-9_@-]+'))
                time.sleep(10)
                browser.get(row[1]+'/page/'+str(i))
            for post in posts:
                time.sleep(10)
                browser.get(post)
                with open('files/tumblr_{0}_page.txt'.format(row[0]), 'w') as content_file:
                    src = browser.page_source
                    html = BeautifulSoup(src, 'lxml')
                    for p in html.find_all('p'):
                        content_file.write(p.getText() + '\n')

crawl_username()
