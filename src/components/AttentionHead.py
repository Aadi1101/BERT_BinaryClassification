from src.logger import logging
from src.exception import CustomException
import sys
import torch
import torch.nn as nn
import torch.nn.functional as f

class AttentionHead(nn.Module):
    try:
        def __init__(self,dim_inp,dim_out):
            super(AttentionHead,self).__init__()
            self.dim_inp = dim_inp
            self.q = nn.Linear(dim_inp,dim_out)
            self.k = nn.Linear(dim_inp,dim_out)
            self.v = nn.Linear(dim_inp,dim_out)

        def forward(self,input_tensor:torch.Tensor,attention_mask:torch.Tensor = None):
            query,key,value = self.q(input_tensor),self.k(input_tensor),self.v(input_tensor)
            scale = query.size(1) ** 0.5
            scores = torch.bmm(query,key.transpose(1,2)) / scale

            scores = scores.masked_fill_(attention_mask,-1e9)
            attn = f.softmax(scores,dim=-1)
            context = torch.bmm(attn,value)
            return context
    except Exception as e:
        raise CustomException(e,sys)