'''
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of BSD 3-Clause License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3-Clause License for more details.
'''
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import math
from torch.nn.modules.utils import _single, _pair, _triple
from torch.utils.cpp_extension import load
adder_cuda = load(
  'adder_cuda', ['adder/adder_cuda.cpp', 'adder/adder_cuda_kernel.cu'], verbose=True)

def adder2d_function(X, W, kernel_size, stride, padding=0):

    n_filters, d_filter, h_filter, w_filter = W.size()  # (n,c,h,w)
    n_x, d_x, h_x, w_x = X.size()
    # print(W.shape)
    # print(X.shape)
    
    stride_w = stride[0]
    stride_h = stride[1]
    h_out = (h_x - h_filter + 2 * padding[0]) / stride_h + 1    # 计算h_out
    w_out = (w_x - w_filter + 2 * padding[1]) / stride_w + 1    # 计算w_out
    # print(h_out)
    # print(w_out)

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), kernel_size, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)   
    #(1,n*c,h,w) -> (1,n*c*h_filter*w_filter,(h-h_filter+1)*(w-w_filter+1)) ->  (n, c*h_filter*w_filter, h_out*w_out)
    X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)
    #(n, c*h_filter*w_filter, h_out*w_out) -> (c*h_filter*w_filter, h_out*w_out, n) -> (h_out*w_out, n*c*h_filter*w_filter)
    W_col = W.view(n_filters, -1)
    #(n_filters, d_filter, h_filter, w_filter) -> (n_filters, d_filter*h_filter*w_filter)
    out = adder.apply(W_col,X_col)
    
    out = out.view(n_filters, h_out, w_out, n_x)  
    out = out.permute(3, 0, 1, 2).contiguous()
    
    return out

class adder(Function):
    @staticmethod
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col,X_col) # 保存反向传播参数
        output = -(W_col.unsqueeze(2)-X_col.unsqueeze(0)).abs().sum(1) # 对应论文公式2
        #-((n_filters, d_filter*h_filter*w_filter, 1) - (1,n, c*h_filter*w_filter, h_out*w_out))
        return output

    @staticmethod
    def backward(ctx,grad_output):
        W_col,X_col = ctx.saved_tensors # 得到反向传播参数
        grad_W_col = ((X_col.unsqueeze(0)-W_col.unsqueeze(2))*grad_output.unsqueeze(1)).sum(2)  
        grad_W_col = grad_W_col/grad_W_col.norm(p=2).clamp(min=1e-12)*math.sqrt(W_col.size(1)*W_col.size(0))/5 # 对应论文公式5
        grad_X_col = (-(X_col.unsqueeze(0)-W_col.unsqueeze(2)).clamp(-1,1)*grad_output.unsqueeze(1)).sum(0)    # 对应论文公式6
        
        return grad_W_col, grad_X_col
    
class adder2d(nn.Module):

    def __init__(self,input_channel,output_channel,kernel_size, stride, padding=0, bias = False,):
        super(adder2d, self).__init__()
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = _pair(kernel_size)
        kernel_size_h = kernel_size[0]
        kernel_size_w = kernel_size[1]
        # print(kernel_size_w)
        # print(kernel_size_h)
        self.adder = torch.nn.Parameter(nn.init.normal_(torch.randn(output_channel,input_channel,int(kernel_size_h),int(kernel_size_w))))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = adder2d_function(x,self.adder, self.kernel_size, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        return output
    
    
