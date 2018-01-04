import numpy as np
import subprocess

word_hidden_layer = np.array([70, 100, 150, 180])

dropout = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
word_embedding = np.array([70, 80, 100, 120 , 150, 200])




for test_no in range(100) :
    config = { 'word_h' : str(word_hidden_layer[np.random.randint(4,size=1)[0]]),
            
            'dropout' : str(dropout[np.random.randint(5,size=1)[0]]),
            'w_embedding' : str(word_embedding[np.random.randint(5,size=1)[0]])}

    for it in range(5):
        print('test_{0}_{1}_{2}_{3}_{4}_{5}.txt iteration {6}'.format(test_no, '20', '2000', '11',config['word_h'], config['w_embedding'], it))
        with open('tune_lstm/test_{0}_{1}_{2}_{3}_{4}_{5}.txt'.format(test_no,'20', '2000', '11' ,config['word_h'], config['w_embedding']), 'a+', encoding='utf-8') as f:
            output = subprocess.check_output(['python','classifier.py', '20', '2000', '11', config['word_h'], config['w_embedding']])
            f.write(output.decode('utf8'))
            f.write('\n\n\n')
    print('--------- next config ---------')

    






