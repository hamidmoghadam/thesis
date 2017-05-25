import csv

n = 0
flag = 0
with open('username_pairs.csv', 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        if flag == 0:
            if -1 == int(row[3]):
                flag = 1
            print(row)
            n += 1
        if flag == 1:
            if (int(row[3])/ len(row[0])) > 0.35:
                print(row)
                n += 1

print(n)