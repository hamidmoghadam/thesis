import pandas as pd
import numpy as np
import sys
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize


def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    #raw_review = BeautifulSoup(raw_review).get_text()
    #
    # 1.1 Remove Url
    GRUBER_URLINTEXT_PAT = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))'

    raw_review = re.sub(GRUBER_URLINTEXT_PAT, "<url>", raw_review)

    # 2. Remove number
    raw_review = re.sub("\d+", "<digit>", raw_review)
    #letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    #
    # 3. Convert to lower case, split into individual words
    #words = raw_review.lower().split()
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(raw_review)
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 5.1 stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(w) for w in meaningful_words]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.

    return(" ".join(stemmed_words))


def main(network, username):
    count = 0
    if network == 'twitter':
        data = pd.read_csv('../tumblr_twitter_scrapper/tweets/' +
                           username + ".csv", sep=' ', header=None)
        data_set = data.as_matrix()
        for row in data_set:
            if row[3] == True:
                print(review_to_words(row[1]))
                count += 1

    if network == 'tumblr':
        data = pd.read_csv('../tumblr_twitter_scrapper/posts/' +
                           username + ".csv", sep=' ', header=None)
        data_set = data.as_matrix()
        for row in data_set:
            if row[5] == True:
                print(review_to_words(str(row[4])))
                count += 1

    print(count)


def sample_from_posts():
    with open('common_corpus.txt', 'w') as f:
        data = pd.read_csv(
            '../tumblr_twitter_scrapper/posts/all.csv', sep=' ', header=None)

        sentences = []
        for post in data.as_matrix():
            if post[5] == True:
                post_content = str(post[4])
                sent_tokenize_list = sent_tokenize(post_content)
                for sent in sent_tokenize_list:
                    txt = review_to_words(sent)
                    if len(txt) > 5:
                        sentences.append(txt)

        #sample_sent = np.random.choice(sentences, size=250)

        #data_set = np.array([review_to_words(str(x[4])) for x in data.as_matrix() if x[5] == True])
        #sample_posts = np.random.choice(data_set, size=100)
        f.write('.\n'.join(sentences))


#main('twitter', 'truthdogg')
sample_from_posts()
