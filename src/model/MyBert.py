import torch
from transformers import XLNetForSequenceClassification, XLNetConfig
from torch.nn import Module
from torch import nn
import NEZHA.utils as utils
import math
from NEZHA.modeling_nezha import BertModel,BertConfig,BertForMaskedLM
from transformers import BertForMaskedLM as RealBertForMaskedLM
# from transformers import BertModel as RealBertModel
SEED = 1997
torch.manual_seed(SEED)
    # 为当前GPU设置种子用于生成随机数，以使结果是确定的
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class MyBert_V1(Module):
    def __init__(self,hidden_size = 768):
        super(MyBert_V1,self).__init__()
        self.pooler = nn.Linear(hidden_size,hidden_size)
        self.Tanh = nn.Tanh()
        self.classification = nn.Linear(hidden_size,2)
        self.dropout = nn.Dropout(0.2)
        self.config = BertConfig("NEZHA/bert_config.json")
        self.pretrained_model = BertModel(config=self.config)
        utils.torch_init_model(self.pretrained_model,'NEZHA/pytorch_model.bin')
    def forward(self,input_ids,attention_mask):
        output = self.pretrained_model(input_ids,attention_mask=attention_mask)[0]
        pooled_output = self.pooler(output)
        pooled_output = self.Tanh(pooled_output)
        pooled_output = torch.mean(pooled_output,1)
        pooled_output = self.dropout(pooled_output)
        output = self.classification(pooled_output)
        return output
    
    
class MyBert_V2(Module):
    def __init__(self,hidden_size = 768,model_name = "nezha-base-cn"):
        super(MyBert_V2,self).__init__()
        self.model_name = model_name
        if model_name == "bert-base-chinese":
            self.pretrained_model = RealBertForMaskedLM.from_pretrained("bert-base-chinese")
        elif model_name == "nezha-base-cn":
            self.config = BertConfig("NEZHA/bert_config.json")
            self.pretrained_model = BertForMaskedLM(config=self.config)
            utils.torch_init_model(self.pretrained_model,'NEZHA/pytorch_model.bin')
        else:
            raise NameError("未找到该模型，请检查或配置该模型")
    def forward(self,input_ids,attention_mask):
        if self.model_name == "bert-base-chinese":
            prediction_scores = self.pretrained_model(input_ids,attention_mask=attention_mask)[0]
        else:
            prediction_scores = self.pretrained_model(input_ids,attention_mask=attention_mask)
        output = prediction_scores.view(-1, 21128)
        return output
    
class MyBert_V3(Module):
    def __init__(self,hidden_size = 768):
        super(MyBert_V3,self).__init__()
        self.config = BertConfig("NEZHA/bert_config.json")
        self.dropout = nn.Dropout(0.2)
        self.pretrained_model = BertForMaskedLM(config=self.config)
        #print(list(self.pretrained_model.named_parameters()))
        #self.pretrained_model.load_state_dict(torch.load('NEZHA/pytorch_model.bin'))
        utils.torch_init_model(self.pretrained_model,'NEZHA/pytorch_model.bin')
        
    def forward(self,input_ids,attention_mask,masked_index):
        prediction_scores = self.pretrained_model(input_ids,attention_mask=attention_mask)
        output = prediction_scores.view(-1,512, self.config.vocab_size)
        if masked_index.shape[0] != 1:
            output = output[:,int(masked_index[0]),:]
        else:
            output = output[:,int(masked_index),:]
        return output

class MyTransformer1(nn.Module):
    def __init__(self, max_len=512, hidden_size=768):
        super(MyTransformer1, self).__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.head_nums = 8
        self.fc_1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(-1)
        self.gelu = nn.GELU()
        self.self_out = self.BertSelfOutput(hidden_size)
        self.out = self.BertOutput(hidden_size)
        self.intermidate = self.BertIntermediate(hidden_size)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        Q = self.q(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 3, 1)
        K = self.k(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 3, 1)
        V = self.v(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 3, 1)
        attention = torch.matmul(
            self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_size // self.head_nums)),
            V).transpose(-2, -1)
        attention = attention.transpose(1, 2)
        attention = attention.reshape(-1, self.max_len, self.hidden_size)
        attention_out = self.self_out(attention,x)
        attention_out = self.intermidate(attention_out)
        attention_out = self.out(attention_out,x)
        out = torch.mean(attention_out, 1)
        out = self.fc_1(out)
        out = self.Tanh(out)
        return out

    class BertSelfOutput(nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-8)
            self.dropout = nn.Dropout(0.2)

        def forward(self, hidden_states, input_tensor):
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states

    class BertIntermediate(nn.Module):
        def __init__(self, hidden_size = 768):
            super().__init__()
            self.dense = nn.Linear(hidden_size, hidden_size * 4)
            self.intermediate_act_fn = nn.GELU()

        def forward(self, hidden_states):
            hidden_states = self.dense(hidden_states)
            hidden_states = self.intermediate_act_fn(hidden_states)
            return hidden_states

    class BertOutput(nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.dense = nn.Linear(hidden_size * 4, hidden_size)
            self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-8)
            self.dropout = nn.Dropout(0.2)

        def forward(self, hidden_states, input_tensor):
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states

class MyTransformer2(nn.Module):
    def __init__(self, max_len=512, hidden_size=768):
        super(MyTransformer2, self).__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.head_nums = 8
        self.fc_1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(-1)
        self.gelu = nn.GELU()
        self.self_out = self.BertSelfOutput(hidden_size)
        self.out = self.BertOutput(hidden_size)
        self.intermidate = self.BertIntermediate(hidden_size)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        Q = self.q(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 3, 1)
        K = self.k(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 3, 1)
        V = self.v(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 3, 1)
        attention = torch.matmul(
            self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_size // self.head_nums)),
            V).transpose(-2, -1)
        attention = attention.transpose(1, 2)
        attention = attention.reshape(-1, self.max_len, self.hidden_size)
        attention_out = self.self_out(attention,x)
        attention_out = self.intermidate(attention_out)
        attention_out = self.out(attention_out,x)
        out = torch.mean(attention_out, 1)
        out = self.fc_1(out)
        out = self.Tanh(out)
        return out

    class BertSelfOutput(nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-8)
            self.dropout = nn.Dropout(0.2)

        def forward(self, hidden_states, input_tensor):
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states

    class BertIntermediate(nn.Module):
        def __init__(self, hidden_size = 768):
            super().__init__()
            self.dense = nn.Linear(hidden_size, hidden_size * 4)
            self.intermediate_act_fn = nn.GELU()

        def forward(self, hidden_states):
            hidden_states = self.dense(hidden_states)
            hidden_states = self.intermediate_act_fn(hidden_states)
            return hidden_states

    class BertOutput(nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.dense = nn.Linear(hidden_size * 4, hidden_size)
            self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-8)
            self.dropout = nn.Dropout(0.2)
            self.act_fn = nn.GELU()
            self.w_u = nn.Linear(hidden_size,hidden_size)
            self.w_o = nn.Linear(hidden_size,hidden_size)

        def forward(self, hidden_states, input_tensor):
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            hidden_gelu = self.act_fn(hidden_states)
            hidden_linear = self.w_u(hidden_states)
            hidden_states = hidden_gelu * hidden_linear
            hidden_states = self.w_o(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states

class MyXLNet(nn.Module):
    def __init__(self,num_labels = 2):
        super(MyXLNet,self).__init__()
        self.config = XLNetConfig.from_pretrained("hfl/chinese-xlnet-base")
        self.config.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.bert = XLNetForSequenceClassification.from_pretrained("hfl/chinese-xlnet-base",config = self.config)
    def forward(self,input_ids,attention_mask):
        y = self.bert(input_ids,attention_mask=attention_mask)[0]
        y = self.dropout(y)
        return y

