import math
import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle

class layer_norm(nn.Module):
    def __init__(self,n_f,eps=1e-9):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_f))
        self.beta = nn.Parameter(torch.zeros(n_f))
    
    def forward(self,inputs):
        mean = torch.mean(inputs,-1,keepdim=True)
        std = torch.mean(inputs,-1,keepdim=True)
        nm = (inputs - mean)/(std + self.eps)
        return (nm * self.gamma) + self.beta
    
class scaled_dot_product_attention(nn.Module):
    def __init__(self,dk=dk,dv=dv): # multi = True tells if the input to the layer will be multi headed or not
        super().__init__()
        self.dk = dk
        self.dv = dv
        
    def forward(self,q,k,v):
        return torch.matmul(F.softmax(torch.matmul(q,k.permute(0,2,1))/math.sqrt(self.dk),2),v)     
        
# utility module

class linear_activated(nn.Module):
    def __init__(self,in_out, activation = nn.ReLU, split = None):
        super().__init__()
        self.linear = nn.Linear(*in_out)
        self.activation = activation()
        self.split = split
        
    def forward(self,inputs):
        if self.split is not None:
            return self.activation(self.linear(inputs)).chunk(self.split,dim=2)
        else:
            return self.activation(self.linear(inputs))
        
class multi_head_attention(nn.Module):
    
    def __init__(self,h=h, dmodel=dmodel, dk=dk, dv=dv, self_attention=False, masked = False):
        super().__init__()
        self.self_attention = self_attention
        # TODO
        self.masked = masked
        
        # method 1 : when dk = dv, and k = v
        # calculate projection of query, key and value at the same time
        if self_attention == True:
            self.linear_kqv = linear_activated((dmodel,dk * h * 3),split=3) 
        else:            
            # method 2
            # second dimension is multiplied by h to calculate all head simultaneously
            self.linear_k = linear_activated((dmodel,dk * h)) 
            self.linear_q = linear_activated((dmodel,dk * h))
            self.linear_v = linear_activated((dmodel,dv * h))
            
        self.linear_o = linear_activated((h*dv,dmodel))
        self.sdpa = scaled_dot_product_attention(dk*h,dv*h)
        
    def forward(self,q,k,v):
        if self.self_attention:
            out_sdpa_multi_head = self.sdpa(*self.linear_kqv(k))
        else:
            out_sdpa_multi_head = self.sdpa(self.linear_k(k),self.linear_q(q),self.linear_v(v))
        return self.linear_o(out_sdpa_multi_head)
    
def get_ffn(dmodel,dff):
    return nn.Sequential(
        linear_activated((dmodel,dff)),
        nn.Linear(dff,dmodel)
        )

class embedding_linear(nn.Module):
    def  __init__(self,vocab_size, dmodel=dmodel, pad=True):
        '''
        Tied weights for decoder embedding layer and pre-softmax linear layer.
        
        vocab_size: size of vocabulary used. It may be different for both source and target
        dmodel: dimension of the word vector
        pad: the pad index in the vocabulary
        '''
        super(embedding_linear,self).__init__()
        self.dmodel = dmodel
        self.weights = nn.Parameter(torch.Tensor(vocab_size,dmodel))
        self.bias = nn.Parameter(torch.Tensor(vocab_size))
        nn.init.xavier_normal_(self.weights.data)
        if pad:
            self.pad_idx = 0
            self.weights.data[0].fill_(0)
        else:
            self.pad_idx = -1
        
        
    def forward(self, inputs, emb=True):
        if emb:
            outputs = F.embedding(inputs, self.weights * (self.dmodel ** 0.5), self.pad_idx, None,2, False, False)
        else:
            outputs = F.linear(inputs,self.weights,self.bias)
        return outputs
            
        
def pos_enc(T=T,dmodel=dmodel):
    # repeats [0;1;...;20](col vec) dmodel times horizonatally to form [[0,0,...,0],[1,1,...,1],...,[19,19,..,19]](20X512)
    pos = torch.arange(0,T).view(-1,1).repeat(1,dmodel)
    # computes the denominator part of the angle inside sin or cos
    denominator_angle_inside_sin_cos = 10000 ** ( 2 * torch.arange(0,dmodel) / dmodel)
    # computes the angle
    encoding = Variable((pos/denominator_angle_inside_sin_cos))
    # encoding even places using sin
    encoding[:, 0::2] = torch.sin(encoding[0:, 0::2]) #  2i
    # encoding odd places with cos
    encoding[:, 1::2] = torch.cos(encoding[0:, 1::2]) # 2i+1
    return encoding

class positional_enc(nn.Module):
    def  __init__(self,T=T, dmodel=dmodel):
        '''
        Tied weights for decoder embedding layer and pre-softmax linear layer.
        
        vocab_size: size of vocabulary used. It may be different for both source and target
        dmodel: dimension of the word vector
        pad: the pad index in the vocabulary
        '''
        super(positional_enc,self).__init__()
        self.embedding = pos_enc(T,dmodel)
        
        
    def forward(self, inputs):
        return self.embedding[inputs]