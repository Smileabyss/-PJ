import time
import numpy as np
from tqdm import tqdm, trange
import random
from args import args
import os
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt


##把并排的训练数据vstack
def data_merge(data, path):
    array = []
    for i in range(data.shape[0]):
        array1 = data[i][0]
        array2 = data[i][1]
        array3 = data[i][2]
        array.append(array1)
        array.append(array2)
        array.append(array3)
    array = np.array(array)
    np.save(path, array)
data = np.load(args.data_path)
data_merge(data, 'data_vstack.npy')


###对loss画图
def loss_painting(path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    checkpoint = torch.load(path)
    losses = checkpoint['loss']
    r = []
    for tensor in losses:
        if type(tensor) == float:
            r.append(tensor)
        else:
            r.append(tensor.detach().numpy())
    # losses = [ tensor.detach().numpy() for tensor in losses]
    losses = np.array(losses)
    # 创建一个二维可视化图
    plt.figure(figsize=(10, 8))
    plt.plot(range(len(losses)), losses,  linestyle='-');

    # 添加图标题和坐标轴标签
    plt.title('model training loss')
    plt.xlabel('epoches')
    plt.ylabel('loss')


    # 显示可视化图
    plt.show()
####数据格式处理
def data_washing_to_jpg(predicted):
   # predicted numpy化   # 数据处理
    predicted = [tensor.detach().cpu().numpy() for tensor in predicted]
    predicted = np.array(predicted)
    predicted = np.squeeze(predicted)
    # print(predicted, type(predicted), predicted.shape)
    predicted = predicted.transpose(1, 2, 0)
    # print(predicted, type(predicted))
    return predicted



#############根据gunet模型
from models import *
def gunet_predict(model_path, save_path, test):
    # if torch.cuda.is_available():
    #     print("CUDA is available! You can use GPU.")
    # else:
    #     print("CUDA is not available. Using CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gunet.gunet_d()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    # model.to(device)

    data = np.load(args.data_path)

    
    for i in test:
        vis = torch.tensor(np.array([data[i][0]]))
        # vis= vis.to(device)

        predicted = model.forward(vis.float())
    ### 数据处理
        predicted = data_washing_to_jpg(predicted)

        path = save_path + str(i) + '.jpg'
        cv2.imwrite(path, predicted)
