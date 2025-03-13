import numpy
import torch
import torch.nn as nn


def linear_test():
    # input size 2，output size 3
    linear_layer = nn.Linear(2, 3)
    print('weight shape', linear_layer.weight.shape, 'bias shape', linear_layer.bias.shape)
    # weight, shape 3 x 2
    linear_layer.weight.data = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
    print(linear_layer.weight.data)
    # print(linear_layer.weight.shape)
    # bias
    linear_layer.bias.data = torch.tensor([1., 2., 3.])
    print(linear_layer.bias.data)
    x = torch.tensor([1., 2.])  # tensor should be float value
    print("input shape: ", x.shape)
    print(x)
    y = linear_layer(x)
    print(y)


def tensor_squeeze():
    a = numpy.array([1, 2, 3])
    t = torch.from_numpy(a)
    ts_0 = t.float().unsqueeze(0)
    ts_1 = t.float().unsqueeze(1)
    print(ts_0)
    print(ts_1)


if __name__ == "__main__":
    tensor_squeeze()
