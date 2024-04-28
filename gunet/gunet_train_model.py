import time
import numpy as np
from tqdm import tqdm, trange
import random
import os
import cv2
import numpy as np
import torch
import torch.optim as optim
import pytorch_ssim
import torch.nn.functional as F
import torch.nn as nn
import gc
from torch.autograd import Variable
from args import args
from models import *


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     # random.seed(seed)
     torch.backends.cudnn.deterministic = True
        

    
def ssim_score(imageA,imageB):
    ssim_loss = pytorch_ssim.SSIM(window_size = 11)
    ##############Confused by ssim loss function, 'pytorch_ssim.py' copy git from https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py####################################
    return ssim_loss(imageA.float(),imageB.float())

def cosine_similarity_loss(A, B):
    # 展平为向量
    A = A.view(A.size(0), -1)
    B = B.view(B.size(0), -1)
    
    # 计算余弦相似度
    cos_sim = F.cosine_similarity(A, B, dim=1)
    
    # 将余弦相似度转换为损失（相似度越高，损失越小）
    loss = 1 - cos_sim.mean()
    
    return loss


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def ssim(self, predicted, target):
        #数据处理
        predicted, target =  predicted.permute(0, 2, 3, 1), target.permute(0, 2, 3, 1)
        # 自定义损失函数的计算逻辑
        loss = 1 - ssim_score(predicted, target)  
        return loss
    def cosloss(self,A, B):
        loss = cosine_similarity_loss(A, B)
        return loss


def train(i, data, path):
    # 手动触发垃圾回收
    gc.collect()
 
    # if torch.cuda.is_available():
    #     print("CUDA is available! You can use GPU.")
    #        # 释放CUDA资源
    #     torch.cuda.empty_cache()
    # else:
    #     print("CUDA is not available. Using CPU.")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将模型移动到GPU上
    model = gunet.gunet_d()
    # model.to(device)

    #加载模型参数
    epoch_init_ = 0
    losses = []  # 用于记录损失值
    if os.path.exists(path):
        
        
        try:
                checkpoint = torch.load(path)
                model.load_state_dict(checkpoint['model'],  strict=False)
                optimizer = optim.Adam(model.parameters(), args.lr)
                optimizer.load_state_dict(checkpoint['optimizer'])
                epoch_init_ = checkpoint['epoch']
                losses = np.load('loss.npy')
                print("模型成功加载")
        except FileNotFoundError:
                print(f"找不到模型文件 {path}，跳过加载模型。")
                 # 定义优化器，
                optimizer = optim.Adam(model.parameters(), args.lr)
                # 如果文件不存在，可以选择使用默认初始化模型或其他备用方法
                # model = YourModelClass()
        except Exception as e:
                 # 定义优化器
                optimizer = optim.Adam(model.parameters(), args.lr)
                print(f"加载模型时发生错误：{e}")
                # 处理其他可能的异常情况
       
   
    else:  optimizer = optim.Adam(model.parameters(), args.lr)

    

    # 定义损失函数
    mse_loss = torch.nn.MSELoss()
    criterion = CustomLoss()
    # criterion.to(device)
    # mse_loss.to(device)

    
    
    length = data.shape[0]
    # 模拟训练过程，每个 epoch 中轮流采样数据
    with tqdm(total=args.epochs-epoch_init_, desc="Training", unit="step") as pbar:
        for epoch in range(epoch_init_, args.epochs):
            # train_data_index =  random.sample(range(length), args.batch_size)
            #随机抽取数据改为轮流抽取数据训练
            train_data_index = [epoch % 1079]#鉴于电脑能承受的batch_size最大为一
            # 将 NumPy 数组转换为 PyTorch 张量
            vis = torch.tensor(np.array([data[i][0] for i in train_data_index]))
            gt = torch.tensor(np.array([data[i][2] for i in train_data_index]))
            # print(type(vis))
            # print(vis.shape, ir.shape, gt.shape)
            # 在每次计算损失之前，将数据传送到GPU
            # vis, gt = vis.to(device), gt.to(device)
            # print(vis.shape, ir.shape, gt.shape)

            outputs = model.forward(vis.float())
            # print(outputs.shape)
            # resolution loss
            outputs, gt = outputs.float(), gt.float()
            ssim_loss_value = criterion.ssim(outputs, gt)
            cosl = criterion.cosloss(outputs, gt)
            pixel_loss_value = 0.
            for i in range(outputs.shape[0]):
                pixel_loss_temp = mse_loss(outputs[i], gt[i])
                pixel_loss_value += pixel_loss_temp
            pixel_loss_value /= outputs.shape[0]
            total_loss =  pixel_loss_value + args.ssim_weight[i]*ssim_loss_value +args.ssim_weight[i]*cosl
            # total_loss =  args.ssim_weight[i]*ssim_loss_value +args.ssim_weight[i]*cosl
            # total_loss = pixel_loss_value + args.ssim_weight[i] * ssim_loss_value

            # 清空之前的梯度，执行反向传播，更新参数
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            losses = np.concatenate([losses, [total_loss.item()]])
            # losses.append(total_loss.item())
            # if len(losses)>1000:
            #     losses = []
            # 更新进度条
            pbar.set_postfix(loss=total_loss, epoches=epoch)
            pbar.update(1)
            if epoch%10 ==0 or epoch == args.epochs-1 :
                #memorize information 
                checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), \
              'epoch': epoch}
                torch.save(checkpoint, path)
                np.save("loss.npy", losses)

                
                

    
if __name__ == "__main__":
    data = np.load(args.data_path)
     # print(data.shape)
    # 设置随机数种子
    setup_seed(args.seed)
    i = 2
    train(i, data, args.model_path)



