#coding=utf-8
import sys, json
sys.path.insert(0,'../..')
from ml.model.sparse_lr import Sparse_LR, Ftrl_LR
from ml.engine.load_data import load_svm

lr_params = {
    'train_file': 'data/a8a.train',
    'test_file': 'data/a8a.test',
    'model_file': 'lr_c0.001.model',
    'learning_rate': {
        'module': 'ml.model.learning_rate',
        'inst': 'DecayLearningRate', #ConstLearningRate, PowerTLearningRate, DecayLearningRate
        'l': 0.01, #const
        'power_t': 0.5,
        'initial': 0.2,
        'decay': 0.999
    },
    'ftrl': {
        'alpha': 0.1,
        'beta': 1,
        'l1': 0.1,
        'l2': 0.1 
    },  
    'use_bias': False,
    'threshold': 0.5,
    'T': 100,
    'bs': 100
}

def train_lr(params):
    lr = Sparse_LR(params)
    lr = Ftrl_LR(params)
    train_y, train_x = load_svm(params['train_file'])
    test_y, test_x = load_svm(params['test_file'])
    
    for t in range(params.get('T',10)):
        batch_size = params.get('bs',100)
        inst_num = 0
        while inst_num < len(train_x):
            lr.update(train_y[inst_num:inst_num+batch_size], train_x[inst_num:inst_num+batch_size])
            inst_num += batch_size
        print 'iteration ', t, 'loss:', lr.loss(train_y, train_x), 'auc:', lr.evaluate(test_y, test_x, metric='auc'), 'aprf:', ','.join(map(str, lr.evaluate(test_y, test_x, metric='f1')))
    lr.save(params['model_file'])
    

if __name__ == '__main__':
    '''
    if len(sys.argv) != 2:
        print 'Usage:\n\tpython lr.py lr.json'
        sys.exit(0)
    train_lr(json.loads(open(sys.argv[1]).read()))
    '''
    train_lr(lr_params)
    
