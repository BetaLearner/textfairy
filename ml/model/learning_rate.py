#coding=utf-8

'''
    "learning_rate": {
        "inst": "ml.model.learning_rate.ConstLearningRate",
        "l": 0.001, #const
        "power_t": 0.5, #power_t
        "initial": 0.5, #decay
        "decay": 0.99   #decay
    }
'''

class LearningRate(object):
    def __init__(self, params):
        self.params = params
        self.l = 0.001

    def get_l(self):
        return self.l


class ConstLearningRate(LearningRate):
    def __init__(self, params):
        super(ConstLearningRate, self).__init__(params)
        self.l = self.params.get('l', 0.001)

    def get_l(self):
        return self.l    

class PowerTLearningRate(LearningRate):
    def __init__(self, params):
        super(PowerTLearningRate, self).__init__(params)
        self.power_t = self.params.get('power_t', 0.5)
        self.t = 1

    def get_l(self):
        self.l = 1.0 / (self.t ** self.power_t)
        self.t += 1
        return self.l

class DecayLearningRate(LearningRate):
    def __init__(self, params):
        super(ExpDecayLearningRate, self).__init__(params)
        self.l = self.params.get('initial', 0.5)
        self.decay = self.params.get('decay', 0.99)

    def get_l(self):
        self.l *= self.decay
        return self.l
