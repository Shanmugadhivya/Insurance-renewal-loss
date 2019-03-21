#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('G:/claim.csv')


# In[3]:


train.head()


# In[4]:


train.info()


# In[5]:


train.describe()


# In[6]:


train.columns


# In[7]:


sns.distplot(train['smoker'])


# In[8]:


train.corr()


# In[9]:


X  = train.iloc[:,1:-1]
Y =  train.iloc[:,6]
X = X.values

Y = Y.values


# In[10]:


imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X)
X= imputer.transform(X)


# In[11]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)


# In[12]:


linear_regression = LinearRegression()

linear_regression.fit(X_train,Y_train)

linear_regression.score(X_train, Y_train)


# In[13]:


prediction = linear_regression.predict(X_test)


# In[20]:


prediction


# In[14]:


linear_regression.predict([[19,0,0,1,3]])


# 

# In[ ]:




