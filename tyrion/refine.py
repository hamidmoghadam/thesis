import os
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



def clean(raw_review, ignore_digit = False, ignore_url= False, ignore_stopword = False):
    # 1. Remove HTML
    #raw_review = BeautifulSoup(raw_review).get_text()
    #
    # 1.1 Remove Url
    if not ignore_url:
        GRUBER_URLINTEXT_PAT = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))'
        raw_review = re.sub(GRUBER_URLINTEXT_PAT, " <url> ", raw_review)
        #insensitive = re.compile(re.escape(url[0]), re.IGNORECASE)
        #parsed_uri = urlparse(url[0])
        #domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
        #raw_review = insensitive.sub(domain, raw_review)

    # 2. Remove number
    if not ignore_digit:
        raw_review = re.sub(" \d+", " <digit> ", raw_review)
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
    sent_tokenize_list = sent_tokenize(txt)
    
    return sent_tokenize_list


