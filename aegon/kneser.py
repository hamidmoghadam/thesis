import kenlm
import csv

lst_user_pair = []
with open('../tumblr_twitter_scrapper/large_username_pairs_filtered.csv', 'r') as username_file:
    reader = csv.reader(username_file, delimiter=' ')

    for row in reader:
        twitter_username = row[2].replace(r'twitter.com/', '')
        tumblr_username = row[0]
        lst_user_pair.append((twitter_username, tumblr_username))

model = kenlm.LanguageModel('../tumblr_twitter_scrapper/tweets/arpa/{0}.arpa'.format(lst_user_pair[1][0]))
for user_pair in lst_user_pair:
    txt = ''
    with open('../tumblr_twitter_scrapper/posts/refined_text/{0}.txt'.format(user_pair[1])) as f:
          txt = f.read()
    
    print('{0} -- {1} : perplexity = {2}'.format(lst_user_pair[1][0], user_pair[1], model.perplexity(txt)))
        