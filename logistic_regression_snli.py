#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# df1=pd.read_json(/home/shubham/Desktop/snli_1.0/snli_1.0/snli_1.0_train.jsonl,lines="true")

# In[3]:




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


# In[4]:






train_datapath = "/home/shubham/Desktop/snli_1.0/snli_1.0/snli_1.0_train.jsonl"
valid_datapath = "/home/shubham/Desktop/snli_1.0/snli_1.0/snli_1.0_dev.jsonl"
test_datapath = "/home/shubham/Desktop/snli_1.0/snli_1.0/snli_1.0_test.jsonl"


df_train = pd.read_json(train_datapath, lines = "true")
df_val = pd.read_json(valid_datapath, lines = "true")
df_test = pd.read_json(test_datapath, lines = "true")


# In[5]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')


# In[6]:


from nltk.stem.lancaster import *
lancasterStemmer =LancasterStemmer()


# In[7]:


stop_words = set(stopwords.words('english'))


# In[8]:



def tokenize(df):
    df['sentence1'] = df['sentence1'].apply(lambda x: [a.lower() for a in x.split(' ')])
    df['sentence2'] = df['sentence2'].apply(lambda x : [a.lower() for a in x.split(' ')])
    
    return df


# In[9]:













# df_train = tokenize(df_train)
# df_val = tokenize(df_val)


# In[ ]:





# In[10]:


df_train=df_train[['sentence1','sentence2','gold_label']]
df_train=df_train.rename(columns={'gold_label':'label'})


# In[11]:



df_val=df_val[['sentence1','sentence2','gold_label']]
df_val=df_val.rename(columns={'gold_label':'label'})
# df_val.head()


# In[12]:


# df_train.head()


# In[13]:


def encode_target(df_train):
    df_train['label'][df_train['label']=='neutral']=0
    df_train['label'][df_train['label']=='entailment']=1
    df_train['label'][df_train['label']=='contradiction']=2
    return df_train


# In[14]:


df_train = encode_target(df_train)
df_val = encode_target(df_val)
df_train.head()

for i in range(len(df_train)):
  if(df_train['label'][i]!=0 and df_train['label'][i]!=1 and df_train['label'][i]!=2):
    df_train['label'][i]=1

for i in range(len(df_val)):
  if(df_val['label'][i]!=0 and df_val['label'][i]!=1 and df_val['label'][i]!=2):
    df_val['label'][i]=1


# In[15]:


# len(df_train['label'])


# In[16]:


# l1=[]
# l2=[]
# for i in range(0,len(df_train)):
#   x=str(df_train['sentence1'][i]+df_train['sentence2'][i])
#   y=str(df_train['label'][i])
#   l1.append(x)
#   l2.append(y)
    
    

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
   
    


# In[17]:


# len(l2)


# In[18]:



from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV


# In[19]:


tfidf=TfidfVectorizer(ngram_range=(1,5), max_features=50000)
model=LogisticRegressionCV()


# In[20]:


x_transfromed =tfidf.fit_transform(l1)


# In[21]:


# print(x_transfromed.shape)


# In[22]:


model.fit(x_transfromed,l2)


# In[23]:


import pickle


# In[24]:


filename = '/home/shubham/Desktop/finalized_model2.sav'
pickle.dump(model, open(filename, 'wb'))


# In[25]:


test_datapath = "/home/shubham/Desktop/snli_1.0/snli_1.0/snli_1.0_test.jsonl"

df_test = pd.read_json(test_datapath, lines = "true")


#df_test=tokenize(df_test)
df_test=df_test[['sentence1','sentence2','gold_label']]
df_test=df_test.rename(columns={'gold_label':'label'})

df_test = encode_target(df_test)

for i in range(len(df_test)):
  if(df_test['label'][i]!=0 and df_test['label'][i]!=1 and df_test['label'][i]!=2):
    df_test['label'][i]=1
# df_test.head()


# In[26]:



# df_test.head()


# In[ ]:





# In[27]:


#df_test=tokenize(df_test)
print("length of test data is ",len(df_test))


# In[28]:




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


# In[29]:


# torch.save(model.state_dict(), filename)


# In[30]:


x_test =tfidf.transform(t1)


# In[31]:


# print(x_test.shape)


# In[65]:


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, t2)
pre=loaded_model.predict(x_test)
print("accuracy: ",result)


# In[41]:


type(pre[0])


# In[34]:


q=model.score(x_test,t2)


# In[40]:


# q


# In[43]:


# pre[:100]


# In[63]:


# saving tfidf in text filed

f=open("tfidf.txt","w")


# In[64]:



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


# In[ ]:





# In[ ]:





# In[ ]:




