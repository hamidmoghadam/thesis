
import re
import numpy as np
from os import listdir
from os.path import isfile, join
import sys

file_adderess = sys.argv[1]
print(file_adderess)
onlyfiles = [f for f in listdir(file_adderess) if isfile(join(file_adderess, f)) and f.startswith('test')]


best_accr = 0.0
best_accr_config = ''

rank_lst = []

for f_name in onlyfiles:
    with open(file_adderess + f_name, 'r') as f:
        expr = r'(?<=post is )[-+]?[0-9]*\.?[0-9]*'
        p = re.compile(expr)
        
        txt = f.read()
        all_accuracies = np.array(re.findall(p, txt), dtype=float)
        ave = np.average(all_accuracies)
        if ave > best_accr:
            best_accr = ave
            best_accr_config = f_name
        rank_lst.append((f_name, round(ave,3)))


rank_lst.sort(key=lambda x: x[1], reverse=True)
print(best_accr)
print(best_accr_config)

print('rank list')
for item in rank_lst:
    print(item)
        
