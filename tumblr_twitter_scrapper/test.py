import csv

lst_large_username = []
with open('large_usernames.csv', 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        lst_large_username.append(row[0])

with open('large_username_pairs_filtered.csv', 'w') as w:
    writer = csv.writer(w, delimiter=' ')
    with open('username_pair_filtered.csv', 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if row[0] in lst_large_username:
                writer.writerow(row)
