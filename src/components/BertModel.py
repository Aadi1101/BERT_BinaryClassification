import sys
from src.logger import logging
from src.exception import CustomException

import torch
import torch.nn as nn
import torch.nn.functional as f
from JointEmbedding import JointEmbedding
from Encoder import Encoder

class BertModel(nn.Module):
    try:
        def __init__(self,vocab_size,dim_inp,dim_out,attention_heads=4):
            super(BertModel,self).__init__()
            self.embedding = JointEmbedding(vocab_size,dim_inp)
            self.encoder = Encoder(dim_inp,dim_out,attention_heads)
            self.token_prediction_layer = nn.Linear(dim_inp,vocab_size)
            self.softmax = nn.LogSoftmax(dim=-1)
            self.classification_layer = nn.Linear(dim_inp,2)

        def forward(self,input_tensor:torch.Tensor,attention_mask:torch.Tensor):
            embedded = self.embedding(input_tensor)
            encoded = self.encoder(embedded,attention_mask)
            token_predictions = self.token_prediction_layer(encoded)
            first_word = encoded[:,0,:]
            return self.softmax(token_predictions),self.classification_layer(first_word)
    except Exception as e:
        raise CustomException(e,sys)