#coding=utf-8
import sys, math
from model import Model
from solver import BaseSolver
#sys.path.insert(0,'../..')
from util.util import get_inst
import ml.evaluate.evaluate
import ml.model.learning_rate

'''
{
    "learning_rate": {
        "module": "ml.model.learning_rate",
        "inst": "ConstLearningRate",
        "l": 0.001, #const
    },
    "threshold": 0.5,
    "model_file": lr_c0.001.model
}
'''
class Sparse_LR(Model):
    def __init__(self, params):
        super(Sparse_LR, self).__init__(params)
        self.model = {}
        self.solver = BaseSolver(self.params)
        self.learning_rate = getattr(sys.modules[self.params['learning_rate']['module']], self.params['learning_rate']['inst'])(self.params['learning_rate'])
        self.threshold = self.params.get("threshold", 0.5)

    def load(self, model_file):
        for line in open(model_file):
            tokens = line.strip().split()
            if len(tokens) != 2:
                print 'Sparse_LR, load model fail, error line:%s', line
            self.model[tokens[0]] = float(tokens[1])

    def save(self, model_file):
        output = open(model_file,'wb')
        for k,v in self.model.items():
            output.write(k + ' ' + str(v) + '\n')
        output.close()

    def score(self, instance):
        wx = 0.0
        for k,v in instance.items():
            wx += self.model.get(k,0.0) * v
        return 1.0/(1.0 + math.exp(-1.0 * wx))

    def predict(self, instances):
        rets = [] 
        for inst in instances:
            rets.append(score(inst))
        return rets

    #lr gradient of neg log likelyhood: (1/(1+e^(-wx))-y)*x
    def update(self, labels, instances):#label
        self.solver.reset()
        for i in range(len(instances)):
            score = self.score(instances[i]) 
            self.solver.cal_gradient(score-labels[i], instances[i])
        for k,v in self.solver.get_gradients().items():
            self.model[k] = self.model.get(k, 0.0) - self.learning_rate.get_l() * v / len(instances)
        
    def loss(self, labels, instances):
        loss = 0
        for i in range(len(instances)):
            score = self.score(instances[i])
            loss += abs(score - labels[i])
            '''
            if 0 == int(labels[i]):
                print 'score', score
                loss += -1.0 * math.log(1 - score)
            elif 1 == int(labels[i]):
                loss += -1.0 * math.log(score)
            else:
                print 'error label', labels[i]
                sys.exit(0)
            '''
        return loss / len(instances)
            
    def evaluate(self, labels, instances, metric='auc'):
        predicts = []
        for i in range(len(instances)):
            predicts.append(self.score(instances[i]))
        return getattr(sys.modules['ml.evaluate.evaluate'], metric)(predicts, labels, self.threshold)

if __name__ == '__main__':
    lr_params = { 
        'train_file': 'data/a8a.train',
        'test_file': 'data/a8a.test',
        'model_file': 'lr_c0.001.model',
        'learning_rate': {
            'module': 'ml.model.learning_rate',
            'inst': 'ConstLearningRate',
            'l': 0.001
        },  
        'threshold': 0.5,
        'T': 10, 
        'bs': 100 
    }
    lr = Sparse_LR(lr_params) 
