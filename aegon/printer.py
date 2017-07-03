from schema import *
import getopt
import csv
import sys
import refine

def provide(network, username):
    dataset_path = ''
    dataItem = None

    if network == 'twitter':
        dataset_path = '../tumblr_twitter_scrapper/tweets/{0}.csv'.format(username)
        dataItem = TwitterItem
    elif network == 'tumblr':
        dataset_path = '../tumblr_twitter_scrapper/posts/{0}.csv'.format(username)
        dataItem = TumblerItem

    result = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            item = dataItem(row)
            if item.isOwner == 'True':
                temp = ' <eos> '.join(refine.get_sentences(item.content))
                result.append(temp)

    return result




def main(argv):

    _network = ''
    _username = ''

    try:
        opts, _ = getopt.getopt(argv, "n:u:")
    except getopt.GetoptError:
        print('print.py -n <network> -u <user>')
    for opt, arg in opts:
        if opt == '-n':
            _network = arg
        elif opt == '-u':
            _username = arg

    for item in provide(_network, _username):
        print(item)

if __name__ == "__main__":
    main(sys.argv[1:])
