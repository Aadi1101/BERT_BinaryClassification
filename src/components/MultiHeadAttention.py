from src.logger import logging
from src.exception import CustomException
import sys
import torch
import torch.nn as nn
import torch.nn.functional as f
from AttentionHead import AttentionHead

class MultiHeadAttention(nn.Module):
    try:
        def __init__(self,num_heads,dim_inp,dim_out):
            super(MultiHeadAttention,self).__init__()
            self.heads = nn.ModuleList([
                AttentionHead(dim_inp,dim_out) for _ in range(num_heads)
            ])
            self.linear = nn.Linear(dim_out * num_heads,dim_inp)
            self.norm = nn.LayerNorm(dim_inp)

        def forward(self,input_tensor: torch.Tensor,attention_mask: torch.Tensor):
            s = [head(input_tensor,attention_mask) for head in self.heads]
            scores = torch.cat(s,dim=-1)
            scores = self.linear(scores)
            return self.norm(scores)
    except Exception as e:
        raise CustomException(e,sys)