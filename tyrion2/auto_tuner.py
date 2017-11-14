import numpy as np
import subprocess

word_hidden_layer = np.array([20, 50, 100, 150])
letter_hidden_layer = np.array([20, 50, 100, 150])
dropout = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
word_embedding = np.array([20,40,50,70, 80])
letter_embedding = np.array([20,40,50,70, 80])



for test_no in range(100) :
    config = { 'word_h' : str(word_hidden_layer[np.random.randint(4,size=1)[0]]),
            'letter_h' : str(letter_hidden_layer[np.random.randint(4,size=1)[0]]),
            'dropout' : str(dropout[np.random.randint(5,size=1)[0]]),
            'w_embedding' : str(word_embedding[np.random.randint(5,size=1)[0]]),
            'l_embedding' : str(letter_embedding[np.random.randint(5,size=1)[0]]) }

    for it in range(25):
        print('test_{0}_{1}_{2}_{3}_{4}_{5}.txt iteration {6}'.format(test_no,config['word_h'], config['w_embedding'], config['letter_h'], config['l_embedding'], config['dropout'].replace('.', '-'), it))
        with open('tune/test_{0}_{1}_{2}_{3}_{4}_{5}.txt'.format(test_no,config['word_h'], config['w_embedding'], config['letter_h'], config['l_embedding'], config['dropout'].replace('.', '-')), 'a+', encoding='utf-8') as f:
            output = subprocess.check_output(['python','rnncnn.py', '10', '1000', '23', config['word_h'], config['w_embedding'], config['letter_h'], config['l_embedding'], '0.0005', config['dropout']])
            f.write(output.decode('utf8'))
            f.write('\n\n\n')
    print('--------- next config ---------')

    






