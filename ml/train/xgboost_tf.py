#coding=utf-8
import sys, math, numpy
from ml.evaluate.evaluate import auc
from ml.model.sparse_lr import Ftrl_LR
from ml.engine.simple_load import load_svm
import xgboost as xgb

def xgboost(base_params, xgboost_params):
    dtrain = xgb.DMatrix(base_params['train_file']) #svm format file
    plst = xgboost_params.items()
    plst += [('eval_metric', 'auc')]
    evallist  = [(dtrain,'train')]
    bst = xgb.train(plst, dtrain, base_params['num_round'], evallist)
    ret = 0 

    if base_params['eval_train']:
        ypred = bst.predict(dtrain)
        ret = auc(dtrain.get_label(), ypred, threshold=0.5)
        print(base_params['train_file'], 'train auc:', ret)
        
    if 'test_file' in base_params and base_params['test_file']:
        dtest = xgb.DMatrix(base_params['test_file'])
        ypred = bst.predict(dtest)
        #ypred = probpred(ypred)
        ret = auc(dtest.get_label(), ypred, threshold=0.5)
        print(base_params['test_file'], 'test auc:', ret)
    bst.save_model(base_params['model_file'])
    return ret 

def get_xgboost_lr_features(labels, pred_leafs, output_file, leaf_num=None):
    if len(labels) != len(pred_leafs):
        print 'get_xgboost_lr_features, len(labels) != len(pred_leafs) error'
        sys.exit(0)
    if not leaf_num:
        leaf_num = numpy.amax(pred_leafs)
        print 'leaf_num:',leaf_num
    output = open(output_file, 'w')
    for idx in range(len(labels)):
        tokens = [str(i*leaf_num + pred_leafs[idx][i]) + ':1' for i in range(len(pred_leafs[idx]))]
        for x in pred_leafs[idx]:
            if x > leaf_num:
                print 'error leaf idx:', pred_leafs[idx]
        output.write(str(int(labels[idx])) + ' ' + ' '.join(tokens) + '\n')
    output.close()
    return leaf_num

def xgboost_lr(base_params, xgboost_params):
    # train xgboost model
    xgboost(base_params, xgboost_params)

    # generate lr feature from xgboost
    bst = xgb.Booster(xgboost_params, model_file=base_params['model_file'])
    dtrain = xgb.DMatrix(base_params['train_file']) #svm format file
    train_y_pred = bst.predict(dtrain, pred_leaf=True)
    leaf_num = get_xgboost_lr_features(dtrain.get_label(), train_y_pred, base_params['xgboost_lr_train_file'])

    dtest = xgb.DMatrix(base_params['test_file'])
    dtest_y_pred = bst.predict(dtest, pred_leaf=True)
    get_xgboost_lr_features(dtest.get_label(), dtest_y_pred, base_params['xgboost_lr_test_file'], leaf_num)

    # train lr model
    lr_params = { 
        'train_file': base_params['xgboost_lr_train_file'],
        'test_file': base_params['xgboost_lr_test_file'],
        'model_file': 'xgboost_lr.model',
        'learning_rate': {
            'module': 'ml.model.learning_rate',
            'inst': 'DecayLearningRate', #ConstLearningRate, PowerTLearningRate, DecayLearningRate
            'l': 0.01, #const
            'power_t': 0.5,
            'initial': 0.2,
            'decay': 0.999
        },  
        'ftrl': {
            'alpha': 0.03,
            'beta': 1,
            'l1': 0.1,
            'l2': 0.1 
        },  
        'use_bias': False,
        'threshold': 0.5,
        'T': 20, 
        'bs': 100 
    }
    lr = Ftrl_LR(lr_params)
    train_y, train_x = load_svm(lr_params['train_file'])
    test_y, test_x = load_svm(lr_params['test_file'])
   
    for t in range(lr_params.get('T',10)):
        batch_size = lr_params.get('bs',100)
        inst_num = 0
        while inst_num < len(train_x):
            lr.update(train_y[inst_num:inst_num+batch_size], train_x[inst_num:inst_num+batch_size])
            inst_num += batch_size
        print 'iteration ', t, \
              'loss:', lr.loss(train_y, [lr.score(x) for x in train_x]), \
              'auc:', lr.evaluate(test_y, [lr.score(x) for x in test_x], metric='auc'), \
              'aprf:', ','.join(map(str, lr.evaluate(test_y, [lr.score(x) for x in test_x], metric='f1')))
        sys.stdout.flush()
    lr.save(lr_params['model_file'])

if __name__ == '__main__':
    base_params={
        'train_file': 'data/a8a.train_',
        'test_file': 'data/a8a.test_',
        'model_file': 'data/xgboost_a8a.model',
        'xgboost_lr_train_file': 'data/a8a.xgboost_lr.train',
        'xgboost_lr_test_file': 'data/a8a.xgboost_lr.test',
        'eval_train': True,
        'num_round': 60
    }
    xgboost_params={
        'bst:eta': 0.7,
        'silent': 1,
        'objective': 'rank:pairwise', #'binary:logistic',
        'bst:max_depth': 3,
        'nthread': 4
    }
    #xgboost(base_params, xgboost_params)
    xgboost_lr(base_params, xgboost_params)
