import numpy as np

#随机梯度下降SGD
class SGD:
    #初始化
    def __init__(self, lr=0.01):
        self.lr = lr
    #参数更新，传入参数字典和梯度字典
    def update(self, params, grads):
        #遍历传入的所有参数，按照公式更新
        for key in params.keys():
            params[key] -= self.lr * grads[key]