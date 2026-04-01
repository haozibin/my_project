import torch
import torch.nn as nn
from torch import device
from torchsummary import summary

# 1. 定义数据
x = torch.randn(10,3)

# 2. 构建模型
model = nn.Sequential(
    nn.Linear(3,4),
    nn.ReLU(),
    nn.Linear(4,4),
    nn.ReLU(),
    nn.Linear(4,2),
    nn.Softmax(dim=1),
)

# 定义一个参数初始化的函数
def init_params(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0.1)

# 3.参数初始化
model.apply(init_params)

# 4.前向传播
out = model(x)
print(out)

summary(model, input_size=(3,),device='cpu')