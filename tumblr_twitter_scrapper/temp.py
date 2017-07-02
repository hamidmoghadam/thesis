import os.path
import csv

with open(r'username_pair_filtered_temp.csv', 'w', encoding='utf-8', newline='') as username_pair_temp:
	writer = csv.writer(username_pair_temp, delimiter = ' ')
	with open(r'username_pair_filtered.csv', 'r', encoding='utf-8') as username_pair:
		reader = csv.reader(username_pair, delimiter=' ')
		lst_username_pair = []
		for row in reader:
			tumblr_username = row[0]
			if os.path.exists('filtered_posts/{0}.csv'.format(tumblr_username)):
				writer.writerow(row)
