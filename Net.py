import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from args import args
import fusion_strategy


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        # 添加输入层到第一个隐藏层的线性层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        if len(hidden_sizes) > 1:
            # 添加多个隐藏层的线性层和ReLU激活函数
            for i in range(len(hidden_sizes) - 1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(nn.ReLU())
        # 添加输出层的线性层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers.append(nn.Sigmoid())
        # 使用nn.Sequential组合所有层
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
###################以下引用https://github.com/hli1221/densefuse-pytorch/blob/master/net.py中网络设计并改动###########

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


# # Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride,out_channels_def):
        super(DenseBlock, self).__init__()
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


##############################################################简单的策略融合网络
class DenseFuse_strategy(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, height=512, width=640, hidden=args.hidden_sizes):
        super(DenseFuse_strategy, self).__init__()
        denseblock = DenseBlock
        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1

        # encoder
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride, out_channels_def = 16)

        # decoder
        self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride)

    def encoder(self, input):
        x1 = self.conv1(input)
        x_DB = self.DB1(x1)
        return x_DB

    def fusion(self, en1, en2, strategy_type='addition'):
        # addition
        if strategy_type == 'attention_weight':
            # attention weight
            fusion_function = fusion_strategy.attention_fusion_weight
        else:
            fusion_function = fusion_strategy.addition_fusion

        f_0 = fusion_function(en1[0], en2[0])
        return f_0

    # def fusion(self, en1, en2, strategy_type='addition'):
    #     f_0 = (en1[0] + en2[0])/2
    #     return [f_0]

    def decoder(self, f_en):
        x2 = self.conv2(f_en)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        output = self.conv5(x4)
        return output


#################################################### 全连接代替策略的一体编码融合解码网络
class DenseFuse_net(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, height = 512, width = 640, hidden = args.hidden_sizes):
        super(DenseFuse_net, self).__init__()
        denseblock = DenseBlock
        nb_filter = [3, 12, 6, 3]
        kernel_size = 3
        stride = 1
        out_channels_def = 3
        # encoder
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride, out_channels_def)

        # function
        self.layers = MultiLayerPerceptron(4*2 * out_channels_def * height * width, hidden,
                                           4*out_channels_def * height * width)

        # decoder
        self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride)
        

    def encoder(self, input):
        x1 = self.conv1(input)
        x_DB = self.DB1(x1)
        return x_DB


    def decoder(self, f_en):
        x2 = self.conv2(f_en)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        output = self.conv5(x4)
        return output
    
    def fusion_net(self, h1, h2):
        # print(h1.shape,h2.shape)
        args = [h1.shape[0], h1.shape[1], h1.shape[2], h1.shape[3]]
        # for i in range(l):
        #     y1 = torch.flatten(h1[i], start_dim=0)
        #     y2 = torch.flatten(h2[i], start_dim=0)
        #     y =  torch.cat((y1, y2))
        #     h.append(y)
        #矩阵扁平化
        h1 = torch.flatten(h1, start_dim=1)
        h2 = torch.flatten(h2, start_dim=1)
        #经过全连接层计算
        h = torch.cat((h1, h2), dim=-1)
        # print(h.shape)
        predicted = self.layers(h)
         #一维张量数据恢复图像矩阵形式
        predicted = predicted.view(args[0], args[1], args[2], args[3])
        return predicted



            
                
    
        
    



