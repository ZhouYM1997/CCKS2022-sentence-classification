from src.model.MyTransformer import MyTransformer1,MyTransformer2
from transformers import BertModel,AutoModel
import torch
import torch.nn as nn
seed = 43
torch.manual_seed(seed)
# 为当前GPU设置种子用于生成随机数，以使结果是确定的
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
"""
transformer_type : 1表示接原始的Transformer，即MyTransformer1，2表示接改进版的Transformer，即MyTransformer2
"""
class Model(nn.Module):
    def __init__(self,max_seq_length = 512,transformer_type = 1,num_labels = 2):
        super(Model, self).__init__()
        self.backbone = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        if transformer_type == 1:
            self.transformer = MyTransformer1(max_len=max_seq_length,hidden_size=312)
        else:
            self.transformer = MyTransformer2(max_len=max_seq_length,hidden_size=312)
        self.fc = nn.Linear(312,num_labels)

    def forward(self,input_ids,attention_mask):
        hidden = self.backbone(input_ids=input_ids,attention_mask=attention_mask)[0]
        intermediate = self.transformer(hidden)
        #output = self.fc(hidden)
        output = self.fc(intermediate)
        output = torch.mean(output,1)
        return output