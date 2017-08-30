#coding=utf-8

class Model(object):
    def __init__(self, params):
        self.params = params

    def load(self, model_path):
        return

    def save(self, model_path):
        return

    def score(self, instance):
        return 

    def predict(self, instances):
        return []

    def update(self, labels, instances):
        return True

    def score_cache(self, data_cache):
        return [], []

    def loss(self, data_cache):
        return

    def loss(self, labels, instances):
        return 
        
    def evaluate(self, data_cache, metric='auc'):
        return

    def evaluate(self, labels, instances, metric='auc'):
        return

    
