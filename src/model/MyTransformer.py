import math
import torch
import torch.nn as nn
seed = 43
torch.manual_seed(seed)
# 为当前GPU设置种子用于生成随机数，以使结果是确定的
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
class MyTransformer1(nn.Module): # Multi-head 版本的Transofrmer layer
    def __init__(self, max_len=512, hidden_size=768):
        super(MyTransformer1, self).__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.head_nums = 8
        self.dropout = nn.Dropout(0.2)
        self.gelu = nn.GELU()
        self.attention = self.Multi_head_attention(head_nums = 8,max_len = max_len,hidden_size = hidden_size)
        self.self_out = self.BertSelfOutput(hidden_size)
        self.out = self.BertOutput(hidden_size)
        self.intermidate = self.BertIntermediate(hidden_size)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        attention = self.attention(x)
        attention_out = self.self_out(attention,x)
        attention_out = self.intermidate(attention_out)
        layer_out = self.out(attention_out, x)
        return layer_out

    class Multi_head_attention(nn.Module):
        def __init__(self, head_nums = 8,max_len = 512,hidden_size = 768):
            super().__init__()
            self.max_len = max_len
            self.head_nums = head_nums
            self.hidden_size = hidden_size
            self.q = nn.Linear(hidden_size, hidden_size)
            self.k = nn.Linear(hidden_size, hidden_size)
            self.v = nn.Linear(hidden_size, hidden_size)
            self.softmax = nn.Softmax(-1)

        def forward(self, x):
            Q = self.q(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 1,
                                                                                                                3)
            K = self.k(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 1,
                                                                                                                3)
            V = self.v(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 1,
                                                                                                                3)
            a = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_size // self.head_nums))
            attention = torch.matmul(
                a,
                V).transpose(-2, -1)
            attention = attention.transpose(1, 2)
            attention = attention.reshape(-1, self.max_len, self.hidden_size)
            return attention

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

class MyTransformer2(nn.Module):# 改进1：用门控机制替换传统FFN；改进2：将做attention时的hidden_size与head_num解绑定
    def __init__(self, max_len=512, hidden_size=768):
        super(MyTransformer2, self).__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.head_nums = 8
        self.dropout = nn.Dropout(0.2)
        self.gelu = nn.GELU()
        self.attention = self.Multi_head_attention(head_nums = 8,max_len = max_len,hidden_size = hidden_size)
        self.self_out = self.BertSelfOutput(hidden_size)
        self.out = self.BertOutput(hidden_size)
        self.intermidate = self.BertIntermediate(hidden_size)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        attention = self.attention(x)
        attention_out = self.self_out(attention,x)
        attention_out = self.intermidate(attention_out)
        layer_out = self.out(attention_out, x)
        return layer_out

    class Multi_head_attention(nn.Module):
        def __init__(self, head_nums = 8,max_len = 512,hidden_size = 768):
            super().__init__()
            self.max_len = max_len
            self.head_nums = head_nums
            self.hidden_size = hidden_size
            self.q = nn.Linear(hidden_size, hidden_size * 4)
            self.k = nn.Linear(hidden_size, hidden_size * 4)
            self.v = nn.Linear(hidden_size, hidden_size)
            self.softmax = nn.Softmax(-1)
            self.relu = nn.ReLU()

        def forward(self, x):

            Q = self.q(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums * 4).permute(0, 2, 1,
                                                                                                                3)
            K = self.k(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums * 4).permute(0, 2, 1,
                                                                                                                3)
            V = self.v(x).reshape(-1, self.max_len, self.head_nums, self.hidden_size // self.head_nums).permute(0, 2, 1,
                                                                                                            3)
            a = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_size // self.head_nums))
            attention = torch.matmul(
                a,
                V).transpose(-2, -1)
            attention = attention.transpose(1, 2)
            attention = attention.reshape(-1, self.max_len, self.hidden_size)
            return attention

    class BertSelfOutput(nn.Module):
        def __init__(self, hidden_size = 768):
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