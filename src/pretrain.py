# coding:utf-8
import numpy as np
import random
import os
import NEZHA.utils as utils
import torch
from torch.utils.data import DataLoader
from model.MyBert import MyBert_V2
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--BATCH_SIZE', type=int, help='BATCH_SIZE', default=8)
parser.add_argument('--EPOCH', type=int, help='EPOCH', default=10)
parser.add_argument('--seed', type=int, help='random seed', default=1997)
parser.add_argument('--max_seq_len', type=int, help='max sequence length', default=512)
parser.add_argument('--lr', type=float, help='learning rate', default=2e-5)
parser.add_argument('--model_name', type=str, help='model_name', default="nezha-base-cn")
args = parser.parse_args()
#设置随机种子
random.seed(args.seed)
np.random.seed(args.seed) 
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

from NEZHA.modeling_nezha import BertModel,BertConfig,BertForMaskedLM
from transformers import Trainer, TrainingArguments,BertTokenizer
from NLP_Utils import MLM_Data,train_data,blockShuffleDataLoader

maxlen=args.max_seq_len
batch_size = args.BATCH_SIZE
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
vocab_size = 21128
train_MLM_data=MLM_Data(train_data,maxlen,tokenizer)

if __name__ == '__main__':
    device = "cuda:0"
    loss_fn = torch.nn.CrossEntropyLoss()
    dataloader = DataLoader(train_MLM_data,batch_size = batch_size,shuffle = True)
    cnt = 0
    model = MyBert_V2()
    model.to(device)
    optim = torch.optim.Adam(lr = args.lr,params = model.parameters())
    model.train()
    for epoch in range(args.EPOCH):
        for item in dataloader:
            optim.zero_grad()
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]
            y_pre = model(input_ids.to(device),attention_mask.to(device)).to(device)
            loss = loss_fn(y_pre.view(-1,vocab_size),labels.view(-1).to(device))
            loss.mean().backward()
            if cnt % 200 == 0:
                print(loss.mean().item())
            optim.step()
            cnt += 1
    
    # 记得新建trained_model这一目录
    torch.save(model.state_dict(),f"trained_model/pretrained_{args.model_name}.pth")
    