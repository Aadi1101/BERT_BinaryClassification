import sys
from src.logger import logging
from src.exception import CustomException

import torch
import torch.nn as nn
import torch.nn.functional as f

class JointEmbedding(nn.Module):
    try:
        def __init__(self,vocab_size,size):
            super(JointEmbedding,self).__init__()
            self.size = size
            self.token_emb = nn.Embedding(vocab_size,size)
            self.segment_emb = nn.Embedding(vocab_size,size)
            self.norm = nn.LayerNorm(size)

        def attention_position(self,dim,input_tensor):
            batch_size = input_tensor.size(0)
            sentence_size = input_tensor.size(-1)
            pos = torch.arange(sentence_size,dtype=torch.long)
            d = torch.arange(dim,dtype=torch.long)
            d = (2*d/dim)

            pos = pos.unsqueeze(1)
            pos = pos / (1e4 ** d)

            pos[:,::2] = torch.sin(pos[:,::2])
            pos[:,1::2] = torch.cos(pos[:,1::2])

            return pos.expand(batch_size,*pos.size())

        def numeric_position(self,dim,input_tensor):
            pos_tensor = torch.arange(dim,dtype=torch.long)
            return pos_tensor.expand_as(input_tensor)  

        def forward(self,input_tensor):
            sentence_size = input_tensor.size(-1)
            pos_tensor = self.attention_position(self.size,input_tensor)
            segment_tensor = torch.zeros_like(input_tensor)
            segment_tensor[:,sentence_size//2+1:] = 1
            output = self.token_emb(input_tensor) + self.segment_emb(segment_tensor) + pos_tensor
            return self.norm(output)
    except Exception as e:
        raise CustomException(e,sys)