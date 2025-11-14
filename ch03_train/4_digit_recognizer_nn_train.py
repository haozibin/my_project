import numpy as np
import matplotlib.pyplot as plt
from two_layer_net import   TwoLayerNet  #两层神经网络
from common.load_data import get_data  #加载数据集函数

#1.加载数据
x_train, t_train, x_test, t_test = get_data()

#2.创建模型
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#3.设置超参数
learning_rate = 0.1
batch_size = 100
num_iters = 10

train_size = x_train.shape[0]
iter_per_epoch = np.ceil(train_size/batch_size)
iter_num = iter_per_epoch * num_iters

train_loss_list = []
train_acc_list = []
test_acc_list = []

#4.循环迭代，用梯度下降法训练模型
for i in range(iter_num):
    #4.1随机选取批量数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    #4.2 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    #4.3 更新参数
    for key in('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]