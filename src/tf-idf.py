#安装相关依赖库 如果是windows系统，cmd命令框中输入pip安装，参考上述环境配置
#!pip install sklearn
#!pip install pandas
#---------------------------------------------------
#导入库
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

#----------------数据探索----------------
# #数据预处理
# #加载训练集
# train_df = pd.read_csv('./data/train.json', sep=',')
# #加载测试集
# test_df = pd.read_csv('./基于论文摘要的文本分类与查询性问答公开数据/test.csv', sep=',')

# #EDA数据探索性分析
# train_df.head()

# test_df.head()

#----------------特征工程----------------
#将Topic(Label)编码
with open('./data/train.json', mode='r', encoding='utf-8') as f:
    text = []
    label = []
    for line in f.readlines():
        json_info = json.loads(line)
        url = json_info['url']
        entity_length = 0
        if "entities" in json_info:
            entity_length = len(json_info['entities'])
        title_length = len(json_info['title'])
        content_length = len(json_info['content'])
        entity_str = ""
        baike_list = []
        for entity in json_info['entities']:
            entity_str += str(json_info['entities'][entity]['co-occurrence'])
            baike_list.extend([str(item['name'])+","+str(item['value']) for item in json_info['entities'][entity]["entity_baike_info"]])
            entity_str += "".join(baike_list)
        entity_str.replace("[","")
        entity_str.replace("]","")
        entity_str.replace(",","")
        entity_str.replace('\"',"")
        text.append(json_info['title'] + json_info['content'] +entity_str)
        label.append(int(json_info["label"]))
    f.close()
train_df = pd.DataFrame()
train_df["Text"] = pd.Series(text)
train_df["Label"] = pd.Series(label)
print(train_df.head())
with open('./data/test_B.json', mode='r', encoding='utf-8') as f:
    text = []
    urls = []
    for line in f.readlines():
        json_info = json.loads(line)
        url = json_info['url']
        urls.append(url)
        entity_length = 0
        if "entities" in json_info:
            entity_length = len(json_info['entities'])
        title_length = len(json_info['title'])
        content_length = len(json_info['content'])
        entity_str = ""
        baike_list = []
        for entity in json_info['entities']:
            entity_str += str(json_info['entities'][entity]['co-occurrence'])
            baike_list.extend([str(item['name'])+","+str(item['value']) for item in json_info['entities'][entity]["entity_baike_info"]])
            entity_str += "".join(baike_list)
        entity_str.replace("[","")
        entity_str.replace("]","")
        entity_str.replace(",","")
        entity_str.replace('\"',"")
        text.append(json_info['title'] + json_info['content'] + entity_str)
    f.close()
test_df = pd.DataFrame()
test_df["Text"] = pd.Series(text)
test_df["url"] = pd.Series(urls)
print(test_df.head())
# train_df['Topic(Label)'], lbl = pd.factorize(train_df['Topic(Label)'])

# #将论文的标题与摘要组合为 text 特征
# train_df['Title'] = train_df['Title'].apply(lambda x: x.strip())
# train_df['Abstract'] = train_df['Abstract'].fillna('').apply(lambda x: x.strip())
# train_df['text'] = train_df['Title'] + ' ' + train_df['Abstract']
# train_df['text'] = train_df['text'].str.lower()

# test_df['Title'] = test_df['Title'].apply(lambda x: x.strip())
# test_df['Abstract'] = test_df['Abstract'].fillna('').apply(lambda x: x.strip())
# test_df['text'] = test_df['Title'] + ' ' + test_df['Abstract']
# test_df['text'] = test_df['text'].str.lower()

#使用tfidf算法做文本特征提取
tfidf = TfidfVectorizer(max_features=2500)

#----------------模型训练----------------

train_tfidf = tfidf.fit_transform(train_df['Text'])
clf = SGDClassifier()
cross_val_score(clf, train_tfidf, train_df['Label'], cv=7)

test_tfidf = tfidf.transform(test_df['Text'])
clf = SGDClassifier()
clf.fit(train_tfidf, train_df['Label'])
test_df['Label'] = clf.predict(test_tfidf)

#----------------结果输出----------------
#test_df['Topic(Label)'] = test_df['Topic(Label)'].apply(lambda x: lbl[x])
cnt_0 = 0
cnt_1 = 0
s = ""
for i in range(test_df.shape[0]):
    item = test_df.iloc[i,:]
    dic = dict()
    dic["url"] = item['url']
    dic["label"] = int(item['Label'])
    if dic["label"] == 1:
        cnt_1 += 1
    else:
        cnt_0 += 1
    #                 if x_2[i][7] < 20:
    #                     if dic["label"] != 0:
    #                         dic["label"] = 0
    #                         cnt_0 += 1

    #                 elif x_2[i][7] > 150:
    #                     if dic["label"] != 1:
    #                         cnt_1 += 1
    #                         dic["label"] = 1
    s += json.dumps(dic) + "\n"
print(cnt_0,cnt_1)
with open(f"output/result_tf_idf.txt",mode='w') as f:
    f.write(s)
    f.close()