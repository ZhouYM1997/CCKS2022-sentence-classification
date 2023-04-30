import pandas as pd
import json
import torch
import numpy as np
import random
from transformers import AutoTokenizer,XLNetTokenizer
from torch.utils.data import Dataset,DataLoader
from NEZHA.modeling_nezha import BertConfig
from model import MyBert
SEED = 1997
# 为当前GPU设置种子用于生成随机数，以使结果是确定的
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
random.seed(SEED)


class MyDataset_V1(Dataset):
    def __init__(self,modify_dataset = False,is_training = True,max_seq_length = 512,hidden_size = 768,original_train_path = "data/train.json",original_test_path = "data/test_B.json",train_path = "data/train_V3.txt",test_path = "data/test_B_V3.txt",prompt = False,input_format = "statistics+320+64",model_name = "nezha-base-cn"):
        super(MyDataset_V1,self).__init__()
        self.is_train = is_training
        self.prompt = prompt
        self.max_seq_length = max_seq_length
        self.input_format = input_format
        if modify_dataset == True:
            if is_training:
                self.data_to_csv(original_train_path,train_path)
            else:
                self.data_to_csv(original_test_path,test_path)
        if is_training:
            self.df = self.read_data(train_path)
        else:
            self.df = self.read_data(test_path)
        if model_name == "xlnet-base-cn" :
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-base")
        elif "nezha" in model_name:
            self.config = BertConfig("NEZHA/bert_config.json")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",config = self.config)
        else:
            raise NameError("未找到该模型，请重新输入或配置该模型！")
    def read_data(self,path):
        df = pd.read_csv(path)
        return df
    def __getitem__(self, item):
        x_str = self.df.iloc[item,0]
        statistics = self.df.iloc[item,1:11]
        if not self.is_train:
            x_str = self.df.iloc[item,1]
            statistics = self.df.iloc[item,2:12]
        input_ids,masked_attention,masked_index = self.convert_examples_to_features(x_str,statistics)
        if self.is_train:
            if self.prompt:
                return (input_ids,masked_attention,masked_index),(torch.Tensor(self.df.iloc[item,1:9])),torch.LongTensor([self.df.iloc[item,-1]])
            return (input_ids,masked_attention),(torch.Tensor(self.df.iloc[item,1:9])),torch.LongTensor([self.df.iloc[item,-1]])
        return (self.df.iloc[item,0]),(input_ids,masked_attention),(torch.Tensor(self.df.iloc[item,2:10]))
    def convert_examples_to_features(self,x_str,statistics):
        lis = self.tokenizer.tokenize(x_str)
        if self.prompt:
            masked_index = lis.index("[MASK]")
        else:
            masked_index = 0
        input_ids = self.tokenizer.convert_tokens_to_ids(lis)
        if "statistics" in self.input_format: # 以取对数的形式加入统计信息，加1是为了防止log函数入参为0导致运算错误。
            for item in statistics:
                input_ids.insert(1,int(np.log(item + 1)))
        
        if len(input_ids) > self.max_seq_length: # 对大于max_seq_len的数据进行处理。
            input_ids = input_ids[:self.max_seq_length - 1]
            input_ids.append(102) 
        masked_attention = [1] * len(input_ids)
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += ([0] * padding_length)
        masked_attention += ([0] * padding_length)
        input_ids = torch.LongTensor(input_ids)
        masked_attention = torch.LongTensor(masked_attention)
        return input_ids,masked_attention,masked_index

    def __len__(self):
        return len(self.df)
    def data_to_csv(self,origin,destination):
        data = []
        with open(origin, mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                json_info = json.loads(line)
                url = json_info['url']
                entity_length = 0
                if "entities" in json_info:
                    entity_length = len(json_info['entities']) # 实体的个数
                title_length = len(json_info['title'])
                content_length = len(json_info['content'])
                cnt_1 = json_info['content'].count("，")  # 内容中的逗号数
                cnt_2 = json_info['content'].count("：")  # 内容中的冒号数
                cnt_3 = json_info['content'].count("。")  # 内容中的句号数
                
                cnt_4 = 0
                cnt_5 = 0
                if self.is_train:
                    y = int(json_info['label'])
                    if self.prompt:
                        y = 3221 if y == 1 else 1415
                if self.input_format == "statistics+320+64":
                    x = "[CLS]" + json_info["title"] + "[SEP]" + json_info["content"][:320] + "[SEP]" + json_info["content"][-64:] + "[SEP]"
                elif self.input_format == "statistics+512+128":
                    x = "[CLS]" + json_info["title"] + "[SEP]" + json_info["content"][:512] + "[SEP]" + json_info["content"][-128:] + "[SEP]"
                else:
                    x = "[CLS]" + json_info["title"] + "[SEP]" + json_info["content"][:200] + "[SEP]"
                if self.prompt:
                    x = "[CLS]" + "是否为优质文章：[MASK]。[SEP]" + json_info["title"] + "[SEP]" + json_info["content"][:320] + "[SEP]" 
                if "entities" in json_info:
                    for entity in json_info['entities']:
                        if "entities" in self.input_format:
                            x += entity+"[SEP]"
                        cnt_4 += len(json_info['entities'][entity]["entity_baike_info"])
                        cnt_5 += len(json_info['entities'][entity]['co-occurrence'])

                if self.is_train:
               
                    data.append([x[:self.max_seq_length], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,y])
                    # if y == 1 and random.random() < 0.05:
                    #     data.append([x[:self.max_seq_length], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,y])
                else:
                   
                    data.append([url,x[:self.max_seq_length], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5])
                    #data.append([url,x_2[:512], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,cnt_6,cnt_7])
                    
            f.close()
        df = pd.DataFrame(data)
        #df.columns = ['x','entity_length','title_length','content_length','cnt_1','cnt_2','cnt_3','cnt_4','cnt_5','cnt_6''cnt_7','y']
        df.to_csv(destination,index=None)
        
        
# class Dataset_V4(Dataset):
#     def __init__(self,modify_dataset = False,is_training = True,max_seq_length = 1024,hidden_size = 768,original_train_path = "data/train.json",original_test_path = "data/test_B.json",train_path = "data/train_V3.txt",test_path = "data/test_B_V3.txt",prompt = False):
#         super(Dataset_V4,self).__init__()
#         self.is_train = is_training
#         self.prompt = prompt
#         self.max_seq_length = max_seq_length
#         if modify_dataset == True:
#             if is_training:
#                 self.data_to_csv(original_train_path,train_path)
#             else:
#                 self.data_to_csv(original_test_path,test_path)
#         if is_training:
#             self.df = self.read_data(train_path)
#         else:
#             self.df = self.read_data(test_path)
#         #self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-base")
#         self.config = BertConfig("NEZHA/bert_config.json")
#         self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",config = self.config)
        
#     def read_data(self,path):
#         df = pd.read_csv(path)
#         return df
#     def __getitem__(self, item):
#         x_str = self.df.iloc[item,0]
#         statistics = self.df.iloc[item,1:11]
#         if not self.is_train:
#             x_str = self.df.iloc[item,1]
#             statistics = self.df.iloc[item,2:12]
        
#         input_ids,masked_attention,masked_index = self.convert_examples_to_features(x_str,statistics)
#         if self.is_train:
#             if self.prompt:
#                 return (input_ids,masked_attention,masked_index),(torch.Tensor(self.df.iloc[item,1:9])),torch.LongTensor([self.df.iloc[item,-1]])
#             return (input_ids,masked_attention),(torch.Tensor(self.df.iloc[item,1:9])),torch.LongTensor([self.df.iloc[item,-1]])
#         return (self.df.iloc[item,0]),(input_ids,masked_attention),(torch.Tensor(self.df.iloc[item,2:10]))
#     def convert_examples_to_features(self,x_str,statistics):
#         lis = self.tokenizer.tokenize(x_str)
#         if self.prompt:
#             masked_index = lis.index("[MASK]")
#         else:
#             masked_index = 0
#         input_ids = self.tokenizer.convert_tokens_to_ids(lis)
#         if len(input_ids) > self.max_seq_length:
#             input_ids = input_ids[:self.max_seq_length - 1]
#             input_ids.append(102)
#         masked_attention = [1] * len(input_ids)
#         padding_length = self.max_seq_length - len(input_ids)
#         input_ids += ([0] * padding_length)
#         masked_attention += ([0] * padding_length)
#         input_ids = torch.LongTensor(input_ids)
#         masked_attention = torch.LongTensor(masked_attention)
#         return input_ids,masked_attention,masked_index
#     def __len__(self):
#         return len(self.df)
#     def data_to_csv(self,origin,destination):
#         data = []
#         with open(origin, mode='r', encoding='utf-8') as f:
#             for line in f.readlines():
#                 json_info = json.loads(line)
#                 url = json_info['url']
#                 entity_length = 0
#                 if "entities" in json_info:
#                     entity_length = len(json_info['entities'])
#                 title_length = len(json_info['title'])
#                 content_length = len(json_info['content'])
#                 cnt_1 = json_info['content'].count("，")  # 内容中的逗号数
#                 cnt_2 = json_info['content'].count("：")  # 内容中的冒号数
#                 cnt_3 = json_info['content'].count("。")  # 内容中的句号数
                
#                 cnt_4 = 0
#                 cnt_5 = 0
#                 if self.is_train:
#                     y = int(json_info['label'])
#                     if self.prompt:
#                         y = 3221 if y == 1 else 1415
                    
                    
#                 #x = "[CLS]" + "该文章是否为优质文章：[MASK]。[SEP]" + json_info["title"] + "[SEP]" + json_info["content"][:256] + "[SEP]" + json_info["content"][-64:] + "[SEP]"
#                 x = "[CLS]" + json_info["title"] + "[SEP]" + json_info["content"][:200] + "[SEP]"
#                 #x = "[CLS]" + json_info["title"] + "[SEP]" + json_info["content"][:320] + "[SEP]" + json_info["content"][-64:] + "[SEP]"
#                 if self.prompt:
#                     x = "[CLS]" + "是否为优质文章：[MASK]。[SEP]" + json_info["title"] + "[SEP]" + json_info["content"][:320] + "[SEP]" 
#                 if "entities" in json_info:
#                     for entity in json_info['entities']:
#                         x += entity+"[SEP]"
#                         cnt_4 += len(json_info['entities'][entity]["entity_baike_info"])
#                         cnt_5 += len(json_info['entities'][entity]['co-occurrence'])
                    
# #                 x = ["","","",""]
                
# #                 idx = 0
# #                 x[idx] = "[CLS]" + json_info["title"] + "[SEP]" + json_info["content"][:160] + json_info["content"][-40:] + "[SEP]"
# #                 for entity in json_info['entities']:
# #                     cnt_4 += len(json_info['entities'][entity]["entity_baike_info"])
# #                     cnt_5 += len(json_info['entities'][entity]['co-occurrence'])
# #                     x[idx] +=  entity +"。"
# #                     x[idx] += "相关词：" + ','.join(json_info['entities'][entity]['co-occurrence'])
# #                     x[idx] += "属性：" + ','.join([val['name'] for val in json_info['entities'][entity]['entity_baike_info'][2:7]]) + "[SEP]"
# #                     if len(x[idx]) > self.max_seq_length:
# #                         temp = x[idx][self.max_seq_length - 20:]
# #                         x[idx] = x[idx][:self.max_seq_length]
# #                         idx += 1
# #                         if idx == 4:
# #                             break
# #                         x[idx] = temp
                    
# #                 while idx < 4:
# #                     if idx != 0:
# #                         x[idx] += x[idx - 1][-100:]
# #                     idx += 1
#                 if self.is_train:
#                     # data.append([x[0][:self.max_seq_length], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,y])
#                     # data.append([x[1][:self.max_seq_length], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,y])
#                     # data.append([x[2][:self.max_seq_length], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,y])
#                     # data.append([x[3][:self.max_seq_length], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,y])
#                     data.append([x[:self.max_seq_length], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,y])
#                     # if y == 1 and random.random() < 0.05:
#                     #     data.append([x[:self.max_seq_length], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,y])
#                 else:
#                     # data.append([url,x[0][:self.max_seq_length], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5])
#                     # data.append([url,x[1][:self.max_seq_length], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5])
#                     # data.append([url,x[2][:self.max_seq_length], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5])
#                     # data.append([url,x[3][:self.max_seq_length], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5])
#                     data.append([url,x[:self.max_seq_length], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5])
#                     #data.append([url,x_2[:512], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,cnt_6,cnt_7])
                    
#             f.close()
#         df = pd.DataFrame(data)
#         #df.columns = ['x','entity_length','title_length','content_length','cnt_1','cnt_2','cnt_3','cnt_4','cnt_5','cnt_6''cnt_7','y']
#         df.to_csv(destination,index=None)