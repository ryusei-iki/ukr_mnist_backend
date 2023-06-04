# import torch
from ukr import ukr
# import matplotlib.pyplot as plt
# # torch.set_default_tensor_type(torch.cuda.FloatTensor)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import datasets, transforms
import numpy as np
import math
from PIL import Image
# 学習データの設定

x_type = 'mnist'


history = {}


if (x_type == 'kura'):
    pass

elif (x_type == 'mnist'):
    def loader(image_path):
        # 画像ファイルを開く
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            gray_img = img.convert('L')
            # RGB形式に変換する
        return gray_img


    # numbers = [892, 892, 892, 892, 892, 892, 892, 892, 892, 892]
    numbers = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    data_num = sum(numbers)
    x = np.zeros((sum(numbers), 28 * 28))
    labels = np.zeros(sum(numbers))
    for i in range(len(numbers)):
        for j in range(numbers[i]):
            # print(sum(numbers[:i]) + j)
            # x[sum(numbers[:i]) + j] = np.array(Image.open('datasets/MNIST/test/{}/{}.png'.format(i, j)))
            x[sum(numbers[:i]) + j] = np.array(Image.open('datasets/MNIST/test/{}/{}.png'.format(i, j))).reshape(-1)
            x[sum(numbers[:i]) + j] = x[sum(numbers[:i]) + j] / 255
            labels[sum(numbers[:i]) + j] = i
# 学習パラメータの設定
eta = 0.1

sigma = (4 * math.log(data_num) / (math.pi * data_num))**(1 / 2)
# sigma = 0.001
epochs = 300
z_dim = 2

z = np.random.uniform(-sigma * 0.01, sigma * 0.01, size=(data_num, z_dim))

ramuda = [1, 30]

np.save('outputs/labels.npy', labels)

exit()
model = ukr(x, labels, z, eta, ramuda, sigma, epochs)
z = model.train()
np.save('outputs/z.npy', z)
np.save('outputs/sigma.npy', sigma)
# np.save('outputs/labe')
