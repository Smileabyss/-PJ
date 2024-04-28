import time
import numpy as np
from tqdm import tqdm, trange
import random
from args import args
import Net
import os
import cv2
import numpy as np
import torch
import torch.optim as optim
import pytorch_ssim
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
    # losses = [tensor.detach().cpu().numpy() for tensor in losses]
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



#####################根据解码编码+融合策略模型生产图片
def defense_predict(model_path, save_path, test, strategy_type):
    if torch.cuda.is_available():
        print("CUDA is available! You can use GPU.")
    else:
        print("CUDA is not available. Using CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      # 将模型移动到GPU上
    model = Net.DenseFuse_strategy(args.in_channels[0],args.in_channels[0])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    data = np.load(args.data_path)
    for i in test:
#####带入模型计算
        vis = torch.tensor(np.array([data[i][0]]))
        ir =  torch.tensor(np.array([data[i][1]]))
        vis, ir = vis.to(device), ir.to(device)

        _vis = model.encoder(vis.float())
        _ir = model.encoder(ir.float())
        ir_vis = model.fusion(_vis, _ir, strategy_type)

        predicted = model.decoder(ir_vis)

    ##### 数据处理
        predicted = data_washing_to_jpg(predicted)

        path = save_path + str(i) + '.jpg'
        print(path)
        cv2.imwrite(path, predicted)
        
################################编码解码model的效果
def decoder_predict(model_path, save_path, test):
    if torch.cuda.is_available():
        print("CUDA is available! You can use GPU.")
    else:
        print("CUDA is not available. Using CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      # 将模型移动到GPU上
    model = Net.DenseFuse_strategy(args.in_channels[0],args.in_channels[0])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    data = np.load(args.data_path)

    for i in test:
        #####带入模型计算
        vis = torch.tensor(np.array([data[i][0]]))
        vis = vis.to(device)
        _vis = model.encoder(vis.float())
        predicted = model.decoder(_vis)

    
##### predicted numpy化   # 数据处理
        predicted = data_washing_to_jpg(predicted)

        path = save_path + str(i) + '.jpg'
        print(path)
        cv2.imwrite(path, predicted)

###############解码编码加网络模型
def defense_net_predict(model_path, save_path, test):
    # if torch.cuda.is_available():
    #     print("CUDA is available! You can use GPU.")
    # else:
    #     print("CUDA is not available. Using CPU.")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型移动到GPU上
    model = Net.DenseFuse_net(args.in_channels[0], args.in_channels[0])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    # model.to(device)

    data = np.load(args.data_path)
    for i in test:
        #####带入模型计算
        vis = torch.tensor(np.array([data[i][0]]))
        ir = torch.tensor(np.array([data[i][1]]))
        # vis, ir = vis.to(device), ir.to(device)

        _vis = model.encoder(vis.float())
        _ir = model.encoder(ir.float())
        ir_vis = model.fusion_net(_vis, _ir)

        predicted = model.decoder(ir_vis)*255

        ##### 数据处理
        predicted = data_washing_to_jpg(predicted)

        path = save_path + str(i) + '.jpg'
        print(path)
        cv2.imwrite(path, predicted)
