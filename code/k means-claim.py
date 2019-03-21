#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries and Dependencies
# 

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the train and test datasets to create two DataFrames
# 

# In[2]:


train_url = "G:/claim.csv"
train = pd.read_csv(train_url)
test_url = "G:/subsiquent claim1-dataset.csv"
test = pd.read_csv(test_url)


# In[3]:


print("***** Train_Set *****")
print(train.head())
print("\n")
print("***** Test_Set *****")
print(test.head())


# In[23]:


print("***** Train_Set *****")
print(train.describe())
print("\n")
print("***** Test_Set *****")
print(test.describe())


# In[24]:


print(train.columns.values)


# In[25]:


train.isna().head()


# In[26]:


test.isna().head()


# In[27]:


print("*****In the train set*****")
print(train.isna().sum())
print("\n")
print("*****In the test set*****")
print(test.isna().sum())


# In[28]:


train.fillna(train.mean(), inplace=True)


# In[29]:


test.fillna(test.mean(), inplace=True)


# In[30]:


print(train.isna().sum())


# In[31]:


print(test.isna().sum())


# In[32]:


train['insuranceclaim'].head()


# In[33]:


train['age'].head()


# In[35]:


train[['region', 'insuranceclaim']].groupby(['region'], as_index=False).mean().sort_values(by='insuranceclaim', ascending=False)


# In[36]:


train[["smoker", "insuranceclaim"]].groupby(['smoker'], as_index=False).mean().sort_values(by='insuranceclaim', ascending=False)


# In[37]:


g = sns.FacetGrid(train, col='insuranceclaim')
g.map(plt.hist, 'age', bins=10)


# In[38]:


grid = sns.FacetGrid(train, col='insuranceclaim', row='region', size=2.2, aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend();


# In[39]:


train.info()


# In[40]:


X = np.array(train.drop(['insuranceclaim'], 1).astype(float))


# In[41]:


y = np.array(train['insuranceclaim'])


# In[42]:


train.info()


# In[43]:


kmeans = KMeans(n_clusters=2) 
kmeans.fit(X)


# In[44]:


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))


# In[45]:


kmeans = kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')
kmeans.fit(X)


# In[46]:


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))


# In[ ]:




