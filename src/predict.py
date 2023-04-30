from model.MyBert import MyBert_V1,MyXLNet
from MyDataset import MyDataset_V1
import torch
from torch.utils.data import DataLoader
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--BATCH_SIZE', type=int, help='BATCH_SIZE', default=8)
parser.add_argument('--DEVICE', type=str, help='DEVICE', default="cuda:0")
parser.add_argument('--model_name', type=str, help='model_name', default="nezha_base_cn")
parser.add_argument('--class_num', type=int, help='num of different classes', default=2)
parser.add_argument('--max_seq_length', type=int, help='max_seq_length', default=512)
parser.add_argument('--model_list', type=str, nargs='+', help='model_list', default=None)
parser.add_argument('--input_format', type=str,  help='input_format', required = True) # 指输入数据的格式，目前用到的有“statistics+320+64”、“statistics+512+256”、“200+entities”三种。第一种表示输入统计信息、标题、内容前320个字符和内容后64个字符；第二种表示输入统计信息、标题、内容前512个字符和内容后256个字符；最后一种表示输入标题、内容前200个字符和实体名称。
args = parser.parse_args()
DEVICE = args.DEVICE
if __name__ == '__main__':
    dataset = MyDataset_V1(modify_dataset = True,is_training = False,max_seq_length = args.max_seq_length,input_format = args.input_format,model_name = args.model_name)
    dataloader = DataLoader(dataset, batch_size=args.BATCH_SIZE)
    for model_name in args.model_list:
        if args.model_name == "nezha-base-cn":
            model = MyBert_V1()
        elif args.model_name == "xlnet-base-cn":
            model = MyXLNet()
        else:
            raise NameError("未找到该模型，请重新填写或配置模型名！")
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(model_name))
        model.eval()
        s = ""
        cnt_0 = 0
        cnt_1 = 0
        for item in dataloader:
            url = item[0]
            x_1 = item[1]
            x_2 = item[2]
            y_pre = model(x_1[0].to(DEVICE),x_1[1].to(DEVICE))
            logits = list(map(int,list(torch.argmax(y_pre,1))))
            #print(y_pre)
            for i in range(0,len(url)):
                dic = dict()
                dic["url"] = url[i]
                dic["label"] = logits[i]
                if dic["label"] == 1:
                    cnt_1 += 1
                else:
                    cnt_0 += 1
                s += json.dumps(dic) + "\n"
        print(cnt_0,cnt_1)
        output_prefix = model_name.replace(".pth","")
        output_prefix = output_prefix.replace("trained_model/","")
        with open(f"output/result_{output_prefix}.txt",mode='w') as f:
            f.write(s)
            f.close()
        