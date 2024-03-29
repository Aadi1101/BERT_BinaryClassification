import sys
from src.logger import logging
from src.exception import CustomException

import torch
import torch.nn as nn
import torch.nn.functional as f
from MultiHeadAttention import MultiHeadAttention

class Encoder(nn.Module):
    try:
        def __init__(self,dim_inp,dim_out,attention_heads=4,dropout=0.1):
            super(Encoder,self).__init__()

            self.attention = MultiHeadAttention(attention_heads,dim_inp,dim_out)
            self.feed_forward = nn.Sequential(
                nn.Linear(dim_inp,dim_out),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(dim_out,dim_inp),
                nn.Dropout(dropout)
            )
            self.norm = nn.LayerNorm(dim_inp)

        def forward(self,input_tensor:torch.Tensor,attention_mask:torch.Tensor):
            context = self.attention(input_tensor,attention_mask)
            res = self.feed_forward(context)
            return self.norm(res)
    except Exception as e:
        raise CustomException(e,sys)