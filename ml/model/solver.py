#coding=utf-8

class LinearSolver(object):
    def __init__(self, params):
        self.params = params
        self.reset()

    def reset(self):
        self.gradients = {}
        self.bias_gradient = 0.0

    def cal_gradient(self, loss, inst):
        return

    def get_gradients(self):
        return self.gradients


class BaseSolver(LinearSolver):
    def __init__(self, params):
        super(BaseSolver, self).__init__(params)
        self.l1 = self.params.get('l1',0.0)
        self.l2 = self.params.get('l2',0.0)

    def cal_gradient(self, loss, inst):
        for k,v in inst.items():
            self.gradients[k] = self.gradients.get(k,0.0) + loss * v
        self.bias_gradient += loss

        
