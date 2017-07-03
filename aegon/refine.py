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
import getopt
from schema import *


def clean(raw_review, ignore_digit = False, ignore_url= False, ignore_stopword = False):
    # 1. Remove HTML
    #raw_review = BeautifulSoup(raw_review).get_text()
    #
    # 1.1 Remove Url
    if not ignore_url:
        GRUBER_URLINTEXT_PAT = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))'
        raw_review = re.sub(GRUBER_URLINTEXT_PAT, "<url>", raw_review)
    # 2. Remove number
    if not ignore_digit:
        raw_review = re.sub("\d+", "<digit>", raw_review)
    #letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    #
    # 3. Convert to lower case, split into individual words
    words = []   
    words = raw_review.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    if not ignore_stopword:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.

    return(" ".join(words))

def stem(txt):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(txt)
    stemmer = PorterStemmer()
    result = []
    for w in words:
        try:
            result.append(stemmer.stem(w))
        except Exception as e:
            result.append(w)

    return ' '.join(result)


def get_sentences(txt):
    txt = clean(txt)
    sent_tokenize_list = sent_tokenize(txt)
    for sent in sent_tokenize_list:
        yield stem(sent)


def main(argv):
    _threshold = 0
    _ignore_digit = False
    _ignore_url = False
    _ignore_stopword = False
    _ignore_stem = False
    
    try:
        opts,_ = getopt.getopt(argv, "t:n:", ["ignore_digit,ignore_url,ignore_stopword,ignore_stem"])
    except getopt.GetoptError:
        print('refine.py -t <threshold>')

    for opt, arg in opts:
        if opt == '-t':
            _threshold = float(arg)
        elif opt == '-n':
            _network = arg
        elif opt == 'ignore_digit':
            _ignore_digit = True
        elif opt == 'ignore_url':
            _ignore_url = True
        elif opt == 'ignore_stem':
            _ignore_stem = True
        elif opt == 'ignore_stopword':
            _ignore_stopword = True

    content_path = ''
    dest_content_path = ''
    language_model_path = ''
    content_index = -1
    content_owner_index = -1
    lst_username = []

    DataItem = None
    if _network == 'tumblr':
        content_path = '../tumblr_twitter_scrapper/posts/text/{0}.txt'
        dest_content_path = '../tumblr_twitter_scrapper/posts/refined_text/{0}.txt'
        language_model_path = '../tumblr_twitter_scrapper/posts/arpa/{0}.arpa'
        DataItem = TumblerItem
        with open(r'../tumblr_twitter_scrapper/large_username_pairs_filtered.csv', 'r', encoding='utf-8') as username_pair:
            reader = csv.reader(username_pair, delimiter=' ')
            lst_username_pair = []
            for row in reader:
                lst_username.append(row[0])
    elif _network == 'twitter':
        content_path = '../tumblr_twitter_scrapper/tweets/text/{0}.txt'
        dest_content_path = '../tumblr_twitter_scrapper/tweets/refined_txt/{0}.txt'
        language_model_path = '../tumblr_twitter_scrapper/tweets/arpa/{0}.arpa'
        DataItem = TwitterItem
        with open(r'../tumblr_twitter_scrapper/large_username_pairs_filtered.csv', 'r', encoding='utf-8') as username_pair:
            reader = csv.reader(username_pair, delimiter=' ')
            lst_username_pair = []
            for row in reader:
                lst_username.append(row[2].replace(r'twitter.com/', ''))
           
    common_model = kenlm.LanguageModel('common_corpus.arpa')
      
    for username in lst_username:
        try:
            private_model = kenlm.LanguageModel(language_model_path.format(username))

            with open(content_path.format(username), 'r') as content_file:
                content = content_file.read()
                new_content = []
                for sent in content.split('<eos>'):
                    if private_model.perplexity(sent) - common_model.perplexity(sent) > _threshold:
                            new_content.append(sent)
            with open(dest_content_path.format(username), 'w') as dest_content_file:
                dest_content_file.write(' <eos> '.join(new_content))
        except Exception as ex:
            print('fail to refine {0}'.format(username))
if __name__ == "__main__":
   main(sys.argv[1:])
