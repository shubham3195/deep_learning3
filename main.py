#!/usr/bin/env python
# coding: utf-8

# In[1]:


#    For Logistic Regression 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import random
import pdb
random.seed(134)
import pandas as pd
import matplotlib.pyplot as plt


PAD_IDX = 0
UNK_IDX = 1
BATCH_SIZE = 32


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



from collections import Counter

# max_vocab_size = 25000
PAD_IDX = 0
UNK_IDX = 1

def build_vocab(all_tokens):
    token_counter = Counter(all_tokens)
#     print(token_counter)
    vocab = token_counter.keys()
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX 
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token

def read_data(fine_name):
    df = pd.read_csv(fine_name,sep='\t')
    return df




import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

from nltk.stem.lancaster import *
lancasterStemmer =LancasterStemmer()


# In[3]:



train_datapath = "./../data/snli_1.0/snli_1.0/snli_1.0_train.jsonl"
valid_datapath = "./../data/snli_1.0/snli_1.0/snli_1.0_dev.jsonl"
test_datapath = "./../data/snli_1.0/snli_1.0/snli_1.0_test.jsonl"


df_train = pd.read_json(train_datapath, lines = "true")
df_val = pd.read_json(valid_datapath, lines = "true")
df_test = pd.read_json(test_datapath, lines = "true")


# In[4]:


stop_words = set(stopwords.words('english'))

def tokenize(df):
    df['sentence1'] = df['sentence1'].apply(lambda x: [a.lower() for a in x.split(' ')])
    df['sentence2'] = df['sentence2'].apply(lambda x : [a.lower() for a in x.split(' ')])
    
    return df


df_train=df_train[['sentence1','sentence2','gold_label']]
df_train=df_train.rename(columns={'gold_label':'label'})



df_val=df_val[['sentence1','sentence2','gold_label']]
df_val=df_val.rename(columns={'gold_label':'label'})
# df_val.head()

def encode_target(df_train):
    df_train['label'][df_train['label']=='neutral']=0
    df_train['label'][df_train['label']=='entailment']=1
    df_train['label'][df_train['label']=='contradiction']=2
    return df_train


df_train = encode_target(df_train)
df_val = encode_target(df_val)
df_train.head()

for i in range(len(df_train)):
  if(df_train['label'][i]!=0 and df_train['label'][i]!=1 and df_train['label'][i]!=2):
    df_train['label'][i]=1

for i in range(len(df_val)):
  if(df_val['label'][i]!=0 and df_val['label'][i]!=1 and df_val['label'][i]!=2):
    df_val['label'][i]=1
    

s1 = df_train['sentence1'].tolist()
s2 = df_train['sentence2'].tolist()
l1 = []
l2=[]


for i in range(len(s1)):
    tokens=word_tokenize(s1[i] + s2[i])
    sen=[]
    for j in tokens:
        if j not in stop_words:
            sen.append(lancasterStemmer.stem(j))
    l1.append(str(sen))  
    l2.append(str(df_train['label'][i]))
   
 
    

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV

tfidf=TfidfVectorizer(ngram_range=(1,5), max_features=50000)
model=LogisticRegressionCV()

x_transfromed =tfidf.fit_transform(l1)

#model.fit(x_transfromed,l2)


# In[6]:


import pickle
# filename = '/home/shubham/Desktop/finalized_model1.sav'
# pickle.dump(model, open(filename, 'wb'))

test_datapath = "./../data/snli_1.0/snli_1.0/snli_1.0_test.jsonl"

df_test = pd.read_json(test_datapath, lines = "true")


#df_test=tokenize(df_test)
df_test=df_test[['sentence1','sentence2','gold_label']]
df_test=df_test.rename(columns={'gold_label':'label'})

df_test = encode_target(df_test)

for i in range(len(df_test)):
  if(df_test['label'][i]!=0 and df_test['label'][i]!=1 and df_test['label'][i]!=2):
    df_test['label'][i]=1
# df_test.head()


print("length of test data is ",len(df_test))




s1 = df_test['sentence1'].tolist()
s2 = df_test['sentence2'].tolist()
t1 = []
t2=[]


for i in range(len(s1)):
    tokens=word_tokenize(s1[i] + s2[i])
    sen=[]
    for j in tokens:
        if j not in stop_words:
            sen.append(lancasterStemmer.stem(j))
    t1.append(str(sen))  
    t2.append(str(df_test['label'][i]))
    
    
x_test =tfidf.transform(t1)


filename = '/models/finalized_model1.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, t2)
pre=loaded_model.predict(x_test)
print("accuracy: ",result)


f=open("tfidf.txt","w")


for j in range(len(df_test)):
    if(int(pre[j])==0):
        f.write('neutral')
        f.write('\n')
    elif(int(pre[j])==1):
        f.write('entailment')
        f.write('\n')
    elif(int(pre[j])==2):
        f.write('contradiction')
        f.write('\n')
print("tfidf.txt file generated...")


# In[9]:


# for RNN

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import random
import pdb
random.seed(134)
import pandas as pd
import matplotlib.pyplot as plt





# In[ ]:



class RNN_dp(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, num_classes, vocab_size,p_p):
        super(RNN_dp, self).__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
#         self.embedding.weight.data.copy_(torch.from_numpy(loaded_embeddings))
#         self.embedding.weight.requires_grad = False
        self.embedding.from_pretrained(torch.from_numpy(loaded_embeddings).cuda(), freeze = True)
        self.drop = nn.Dropout(p=p_p)


        
        self.rnn = nn.GRU(emb_size, hidden_size, num_layers, batch_first = True)
        self.linear1 = nn.Linear(2*hidden_size, 500)
        self.linear2 = nn.Linear(500,num_classes)

    def init_hidden(self, batch_size):
        hidden = torch.randn(self.num_layers, batch_size, self.hidden_size)
        return hidden.cuda()

    def forward(self, data_s1, length1, data_s2, length2):  
        batch_size = data_s1.size(0)
        self.hidden1 = self.init_hidden(batch_size)
        self.hidden2 = self.init_hidden(batch_size)

        embed1 = self.embedding(data_s1)
        embed2 = self.embedding(data_s2)
        
#         print(embed1.size())
        rnn_out1_, hidden1 = self.rnn(embed1, self.hidden1)
        rnn_out2_, hidden2 = self.rnn(embed2, self.hidden2)
#         print(self.hidden1.size())

        rnn_out1 = torch.sum(hidden1, dim=0)
        rnn_out2 = torch.sum(hidden2, dim=0)
#         print(rnn_out1.size())
        combined_out = torch.cat([rnn_out1, rnn_out2], dim=1)
#         print(combined_out.size())
        logits = F.relu(self.linear1(combined_out))
        res = self.linear2(self.drop(logits))
        
        return res





