import copy
import random
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split
from transformers import AutoTokenizer
import transformers.models.bert.modeling_bert
from torchtext.datasets import AG_NEWS
from src.model.BertV2 import Model
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def fix_seed(seed):
    torch.manual_seed(seed)
    # 为当前GPU设置种子用于生成随机数，以使结果是确定的
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

BATCH_SIZE = 2
DEVICE = "cuda:0"
seed = 43
fix_seed(seed)
max_seq_length = 64
k = 100
train_iter = AG_NEWS(split='train')
train_loader = DataLoader(train_iter,batch_size=BATCH_SIZE)
train_list = list(train_loader)
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
def padding(encoding):
    encoding["position_ids"] = []
    for i in range(len(encoding["input_ids"])):
        padding_length = max_seq_length - len(encoding["input_ids"][i])
        encoding["input_ids"][i] += ([0] * padding_length)
        encoding["token_type_ids"][i] += ([0] * padding_length)
        encoding["attention_mask"][i] += ([0] * padding_length)
        encoding["position_ids"].append([_ for _ in range(max_seq_length)])
    return encoding

#model = TransformerForSequenceClassification(2)
#model = model.to(DEVICE)
model = Model(max_seq_length=max_seq_length,num_labels=5,transformer_type=1).to(DEVICE)
#model.config["num_labels"] = 5
#model.embedding_input_ids.load_state_dict(bert.embeddings.word_embeddings.state_dict())
# model.embedding_position_ids.load_state_dict(bert.embeddings.position_embeddings.state_dict())
# model.embedding_token_type_ids.load_state_dict(bert.embeddings.token_type_embeddings.state_dict())
loss_fn = nn.CrossEntropyLoss()

def get_group_parameters(model):
    params = list(model.named_parameters())
    no_decay = ['bias,','LayerNorm']
    other = ['transformer','fc']
    no_main = no_decay + other
    param_group = [

        {'params':[p for n,p in params if not any(nd in n for nd in no_main)],'weight_decay':1e-3},
        {'params':[p for n,p in params if not any(nd in n for nd in other) and any(nd in n for nd in no_decay) ],'weight_decay':0},
        {'params':[p for n,p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay) ],'weight_decay':0,'lr':5e-4},
        {'params':[p for n,p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay) ],'weight_decay':1e-3,'lr':5e-4},
    ]
    return param_group
parameters = get_group_parameters(model)
lr = 2e-5
optimizer = torch.optim.Adam(parameters,lr = lr)
i = 0
j = 0
cnt = 0
process = "train"
running_loss = 0
total_step = 1000
for i in range(total_step):
    if process == "train":
        idx = np.random.randint(0, len(train_list) - k)
        label, line = train_list[idx]
        optimizer.zero_grad()
        model.zero_grad()
        encoding = tokenizer(list(line),padding=True,max_length=max_seq_length,truncation=True)
        encoding = padding(encoding)
        encoding1 = torch.LongTensor(encoding["input_ids"]).to(DEVICE)
        encoding2 = torch.LongTensor(encoding["attention_mask"]).to(DEVICE)
        #y_pre = model(encoding1, encoding2, encoding3)
        y_pre = model(input_ids=encoding1,attention_mask=encoding2)
        loss = loss_fn(y_pre,torch.LongTensor(label).to(DEVICE))
        loss.mean().backward()
        optimizer.step()
        running_loss += loss.mean().item()
        if i % 50 == 0:
            print(running_loss)
            running_loss = 0
        if i >= total_step - 201:
            process = "dev"
    else:
        idx = np.random.randint(len(train_list) - k, len(train_list))
        label, line = train_list[idx]
        with torch.no_grad():
            encoding = tokenizer(list(line),padding=True,max_length=max_seq_length,truncation=True)
            encoding = padding(encoding)
            encoding1 = torch.LongTensor(encoding["input_ids"]).to(DEVICE)
            encoding2 = torch.LongTensor(encoding["attention_mask"]).to(DEVICE)
            y_pre = model(input_ids=encoding1,attention_mask=encoding2)
            y_pre = np.array(y_pre.cpu())
            y_pre = y_pre.argmax(1)
            print(y_pre)
            for i in range(len(y_pre)):
                if y_pre[i] == label[i]:
                    cnt += 1
print(cnt)







