import pytumblr
import csv
from bs4 import BeautifulSoup
import time

# Authenticate via OAuth
client = pytumblr.TumblrRestClient(
  'cUaEmGIleDG46g7L58aeNhtcwMZlWRLXTPRKYsEPlzybMzHtxV',
  'Ra1jDVcpjLQD81tQwSl4Lk9iSABDDQ7DNfPgedGaMoGI4VP5WW',
  't81wrl8ZQPKnEdhOANppgwl4KkXgoadZtfby3z044BxfTcGhmw',
  'n8vEi9Y5sHRkfNPOfyDnR7lBeeMF2z2S2aV7OmBd2w5HaVBGw9'
)

# Make the request

#print(client.blog_info('reincepriebus')) # get information about a blog

fetched_user_set = set()
with open('user_data_fetched.csv', 'r') as user_data_fetched:
    reader = csv.reader(user_data_fetched, delimiter=' ')
    for row in reader:
        fetched_user_set.add(row[0])

with open('username_pairs.csv', 'r') as username_pairs:
    reader = csv.reader(username_pairs, delimiter= ' ')
    with open('user_data_fetched.csv', 'a') as user_data_fetched:
        write_fetched_user = csv.writer(user_data_fetched, delimiter=' ')
        for row in reader:
            username = row[0]
            if username in fetched_user_set:
                continue
            print('{0} crawling started'.format(username))
            with open('posts/{0}.csv'.format(username), 'a+') as post_file:
                writer = csv.writer(post_file, delimiter=' ')
                try:
                    data = client.posts(username, 'text')
                    for post in data['posts']:
                        id = post['id']
                        trail = post['trail']
                        user_is_owner = True
                        if len(trail) > 0:
                            user_is_owner = trail[0]['post']['id'] == str(id)
                        date = post['date']
                        body = post['body'].replace('\n', ' ')
                        body = BeautifulSoup(body, 'lxml').get_text()
                        url = post['post_url']

                        writer.writerow([id, date, url, body, user_is_owner])

                    data = client.posts(username, 'link')
                    for post in data['posts']:
                        id = post['id']
                        trail = post['trail']
                        user_is_owner = True
                        if len(trail) > 0 :
                            user_is_owner = trail[0]['post']['id'] == str(id)
                        date = post['date']
                        body = post['description'].replace('\n', ' ')
                        body = BeautifulSoup(body, 'lxml').get_text()
                        url = post['post_url']

                        writer.writerow([id, date, url, body, user_is_owner])
                    print('{0} crawling finished'.format(username))
                    write_fetched_user.writerow([username])
                    user_data_fetched.flush()
                except Exception as e:
                    print(e)
            time.sleep(10)
        #pdb.set_trace()



#client.avatar('codingjester') # get the avatar for a blog
#print(client.blog_likes('reincepriebus')) # get the likes on a blog
#print(client.followers('reincepriebus')) # get the followers of a blog
#client.queue('codingjester') # get the queue for a given blog
#client.submission('codingjester') # get the submissions for a given blog