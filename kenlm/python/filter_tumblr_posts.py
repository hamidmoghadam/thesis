import os
import kenlm
import pandas as pd
import numpy as np
import sys
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
import csv


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




with open(r'../../tumblr_twitter_scrapper/username_pair_filtered.csv', 'r', encoding='utf-8') as username_pair:
    reader = csv.reader(username_pair, delimiter=' ')
    lst_username_pair = []
    for row in reader:
        lst_username_pair.append((row[0], row[2].replace(r'twitter.com/', '')))
    
    LM = os.path.join(os.path.dirname(__file__), '..', 'lm', 'common_corpus.arpa')
    model = kenlm.LanguageModel(LM)
                
    for item in lst_username_pair:#[:user_count]:
        tumblr_username = item[0]
        print(tumblr_username)
        posts_path = '../../tumblr_twitter_scrapper/posts/{0}.csv'.format(
            tumblr_username)
        posts_new_path = '../../tumblr_twitter_scrapper/filtered_posts/{0}.csv'.format(tumblr_username)
        data = pd.read_csv(posts_path, sep=' ', header=None)
        sentences = []
        scores = []
        data_set = data.as_matrix()
        for post in data_set:
            if post[5] == True:
                post_content = str(post[4])
                sent_tokenize_list = sent_tokenize(post_content)
                for sent in sent_tokenize_list:
                    txt = review_to_words(sent)
                    if len(txt) > 5:
                        scores.append(model.score(txt))
        
        scores.sort()
        if len(scores) == 0:
            continue
        threashold = scores[len(scores)//2]
        with open(posts_new_path, 'w') as new_post:
            writer = csv.writer(new_post, delimiter = ' ')
            for post in data_set:
                if post[5] == True:
                    new_post = ''
                    post_content = str(post[4])
                    sent_tokenize_list = sent_tokenize(post_content)
                    for sent in sent_tokenize_list:
                        txt = review_to_words(sent)
                        if len(txt) > 5:
                            if model.score(txt) < threashold:
                                new_post +=' ' + txt
                    if len(new_post) > 10:
                        post[4] = new_post
                        writer.writerow(post)

            

'''print('{0}-gram model'.format(model.order))

sentence = 'language modeling is fun .'
print(sentence)
print(model.score(sentence))

# Check that total full score = direct score
def score(s):
    return sum(prob for prob, _ in model.full_scores(s))

assert (abs(score(sentence) - model.score(sentence)) < 1e-3)

# Show scores and n-gram matches
words = ['<s>'] + sentence.split() + ['</s>']
for i, (prob, length) in enumerate(model.full_scores(sentence)):
    print('{0} {1} : {2}'.format(prob, length, ' '.join(words[i+2-length:i+2])))

# Find out-of-vocabulary words
for w in words:
    if not w in model:
        print('"{0}" is an OOV'.format(w))'''
