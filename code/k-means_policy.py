#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_url = "G:/policy.csv"
train = pd.read_csv(train_url)
test_url = "G:/policy1.csv"
test = pd.read_csv(test_url)


# In[3]:


print("***** Train_Set *****")
print(train.head())
print("\n")
print("***** Test_Set *****")
print(test.head())


# In[4]:


print("***** Train_Set *****")
print(train.describe())
print("\n")
print("***** Test_Set *****")
print(test.describe())


# In[5]:


print(train.columns.values)


# In[6]:


train.isna().head()


# In[7]:


test.isna().head()


# In[8]:


print("*****In the train set*****")
print(train.isna().sum())
print("\n")
print("*****In the test set*****")
print(test.isna().sum())


# In[9]:


train.fillna(train.mean(), inplace=True)


# In[10]:


test.fillna(test.mean(), inplace=True)


# In[11]:


print(train.isna().sum())


# In[12]:


print(test.isna().sum())


# In[13]:


train['Year of Purchase'].head()


# In[14]:


train['Age'].head()


# In[16]:


train[['Vehicle Age Years', 'Year of Purchase']].groupby(['Vehicle Age Years'], as_index=False).mean().sort_values(by='Year of Purchase', ascending=False)


# In[17]:


train[['Type of Coverage', 'Year of Purchase']].groupby(['Type of Coverage'], as_index=False).mean().sort_values(by='Year of Purchase', ascending=False)


# In[20]:


g = sns.FacetGrid(train, col='Year of Purchase')
g.map(plt.hist, 'Age', bins=5)


# In[21]:


grid = sns.FacetGrid(train, col='Year of Purchase', row='Vehicle Age Years', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[22]:


train.info()


# In[32]:


labelEncoder = LabelEncoder()
labelEncoder.fit(train['Nationality'])

train['Nationality'] = labelEncoder.transform(train['Nationality'])

labelEncoder = LabelEncoder()
labelEncoder.fit(train['Type of Coverage'])

train['Type of Coverage'] = labelEncoder.transform(train['Type of Coverage'])

labelEncoder = LabelEncoder()
labelEncoder.fit(train['Restricted No. of Drivers'])

train['Restricted No. of Drivers'] = labelEncoder.transform(train['Restricted No. of Drivers'])


# In[33]:


train.info()


# In[34]:


X = np.array(train.drop(['No Claim Availed'], 1).astype(float))


# In[35]:


y = np.array(train['No Claim Availed'])


# In[36]:


train.info()


# In[37]:


kmeans = KMeans(n_clusters=2) 
kmeans.fit(X)


# In[38]:


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))


# In[39]:


kmeans = kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')
kmeans.fit(X)


# In[40]:


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))


# In[ ]:




