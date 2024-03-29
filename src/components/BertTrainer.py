import sys
from src.logger import logging
from src.exception import CustomException
from pathlib import Path

import torch
import time
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from IMDBBertDataset import IMDBBertDataset
from BertModel import BertModel

class BertTrainer:
    try:
        def __init__(self,model:BertModel,dataset:IMDBBertDataset,print_progress_every:int=10,print_accuracy_every:int=50,batch_size:int=24,learning_rate:float=0.005,epochs:int=5):
            self.model = model
            self.dataset = dataset
            self.batch_size = batch_size
            self.epochs = epochs
            self.current_epoch = 0
            self.loader = DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True)
            self.criterion = nn.BCEWithLogitsLoss()
            self.ml_criterion = nn.NLLLoss(ignore_index=0)
            self.optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.015)
            self._splitter_size = 35
            self._ds_len = len(self.dataset)
            self._batched_len = self._ds_len // self.batch_size
            self._print_every = print_progress_every
            self._accuracy_every = print_accuracy_every

        def print_summary(self):
            ds_len = len(self.dataset)
            print("Model Summary\n")
            print("="* self._splitter_size)
            print(f"Training dataset len: {ds_len}")
            print(f"Max/Optimal sentence len: {self.dataset.optimal_sentence_length}")
            print(f"Vocab Size: {len(self.dataset.vocab)}")
            print(f"Batch Size: {self.batch_size}")
            print(f"Batched dataset len: {self._batched_len}")
            print("="* self._splitter_size)
            print()

        def __call__(self):
            for self.current_epoch in range(self.current_epoch,self.epochs):
                loss = self.train(self.current_epoch)

        def train(self,epoch:int):
            print(f"Begin epoch {epoch}")
            prev = time.time()
            average_nsp_loss = 0
            average_mlm_loss = 0
            for i,value in enumerate(self.loader):
                index = i +1
                inp,mask,inverse_token_mask,token_target,nsp_target = value
                self.optimizer.zero_grad()

                token,nsp = self.model(inp,mask)
                tm = inverse_token_mask.unsqueeze(-1).expand_as(token)
                token = token.masked_fill(tm,0)

                loss_taken = self.ml_criterion(token.transpose(1,2),token_target)
                loss_nsp = self.criterion(nsp,nsp_target)

                loss = loss_taken + loss_nsp
                average_nsp_loss += loss_nsp
                average_mlm_loss += loss_taken

                loss.backward()
                self.optimizer.step()

                average_nsp_loss = 0
            average_mlm_loss = 0
            print(f"Loss: {loss}")
            return loss
    except Exception as e:
        raise CustomException(e,sys)
    

if __name__ == '__main__':
    EMB_SIZE = 64
    HIDDEN_SIZE = 36
    EPOCHS = 4
    BATCH_SIZE = 12
    NUM_HEADS = 4
    print("Prepare Dataset")
    BASE_DIR = Path(__file__).resolve().parent
    ds = IMDBBertDataset(BASE_DIR.joinpath('data/imdb.csv'), ds_from=0, ds_to=1000, should_include_text=True)

    bert = BertModel(len(ds.vocab),EMB_SIZE,HIDDEN_SIZE,NUM_HEADS)
    trainer = BertTrainer(
        model=bert,
        dataset=ds,
        print_progress_every=20,
        print_accuracy_every=200,
        batch_size=BATCH_SIZE,
        learning_rate=0.00007,
        epochs=4
    )
    print(trainer.print_summary())
    trainer()