import argparse
import pandas as pd
import json
parser = argparse.ArgumentParser()
parser.add_argument('--dir_name', type=str, help='The name of directory', required = True)
parser.add_argument('--is_filter', type=str, help='Is or not to filter the prediction value which definitely not accord with statistics.', default = False)
args = parser.parse_args()

import os
if __name__ == "__main__":
    file_list = []
    for root, dirs, files in os.walk(args.dir_name):
        for file_name in files:
            file_list.append(args.dir_name + "/" + file_name)
    print(f"参与投票的文件个数为：{len(file_list)}")
    if len(file_list) % 2 == 0:
        raise AssertionError("文件数量不为奇数个，不能参与投票，请检查")
    df = pd.DataFrame()
    with open(f"data/test_B.json",encoding="utf-8", mode = 'r') as f:
        urls = []
        if args.is_filter:
            entities = []
        for line in f.readlines():
            json_info = json.loads(line)
            urls.append(json_info['url'])
            if args.is_filter:
                entities.append(json_info['entities'])
        f.close()
        df['url'] = urls
        if args.is_filter:
            df['entities'] = entities

    for file in file_list:
        with open(file,encoding="utf-8", mode = 'r') as f:
            data = []
            for line in f.readlines():
                json_info = json.loads(line)
                data.append(json_info["label"])
            df[file] = data
            f.close()

    new_column = df[file_list[0]]
    for i in range(1,len(file_list)):
        new_column += df[file_list[i]]
    zero_up = len(file_list) // 2
    one_bottom = zero_up + 1
    new_column[new_column <= zero_up] = 0
    new_column[new_column >= one_bottom] = 1
    df["vote"] = new_column
    
    if args.is_filter:
        def get_vote(x):
            if x["vote"] == 0:
                if len(x['entities']) >= 60:
                    return 1
                else:
                    return 0
            else:
                if len(x['entities']) <= 12:
                    return 0
            return 1
        df["vote"] = df.apply(get_vote,axis=1)
    zero_cnt = df[df["vote"] == 0]["vote"].count()
    one_cnt = df[df["vote"] == 1]["vote"].count()
    print(f"预测结果中0的数量为：{zero_cnt}")
    print(f"预测结果中1的数量为：{one_cnt}")
    print(f"预测结果1的数量占总数据量的占比为：{one_cnt / (zero_cnt + one_cnt) * 100}%")
    
    s = ""
    for i in range(df.shape[0]):
        dic = dict()
        dic['url'] = df.iloc[i,:]["url"]
        dic['label'] = int(df.iloc[i, :]["vote"])
        s += json.dumps(dic,ensure_ascii=False) + "\n"
        
    with open("result.txt",encoding="utf-8", mode='w') as f:
        f.write(s)
        f.close()
