#coding=utf-8
import sys, math, numpy
import ml.evaluate.evaluate
from ml.engine.simple_load import load_svm_as_numpy_array

import lightgbm as lgb
import numpy as np


def lightgbm(base_params, gbm_params):
    # load data
    y_train, x_train, max_feature = load_svm_as_numpy_array(base_params['train_file'], None)
    y_test, x_test, max_feature = load_svm_as_numpy_array(base_params['test_file'], max_feature)
    print('max_feature', max_feature)

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

    # train
    gbm = lgb.train(gbm_params, lgb_train, num_boost_round=100, valid_sets=lgb_train)
    print('train success.')

    # predict and get data on leaves, training data
    y_pred = gbm.predict(x_train,pred_leaf=True)
    print('predict success.') 
    sys.stdout.flush()

    # feature transformation and write result
    print('Writing transformed training data')
    transformed_training_matrix = np.zeros([len(y_pred),len(y_pred[0]) * num_leaf],dtype=np.int64)
    for i in range(0,len(y_pred)):
        temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
        transformed_training_matrix[i][temp] += 1

    y_pred = gbm.predict(X_test,pred_leaf=True)

    print('Feature importances:', list(gbm.feature_importance()))
    print('Feature importances:', list(gbm.feature_importance("gain")))

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
    gbm_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 20,
        'num_trees': 10,
        'learning_rate': 0.01,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    lightgbm(base_params, gbm_params)
