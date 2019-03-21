#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression



get_ipython().run_line_magic('matplotlib', 'inline')


# In[77]:


df = pd.read_csv('G:/policy cancellation1.csv')


# In[48]:


df.head()


# In[49]:


df.info()


# In[50]:


labelEncoder = LabelEncoder()
labelEncoder.fit(df['Nationality'])

df['Nationality'] = labelEncoder.transform(df['Nationality'])


# In[51]:


df.info()


# In[52]:


labelEncoder = LabelEncoder()
labelEncoder.fit(df['Type of Coverage'])

df['Type of Coverage'] = labelEncoder.transform(df['Type of Coverage'])


# In[55]:


df.info()


# In[56]:


labelEncoder = LabelEncoder()
labelEncoder.fit(df['Restricted No. of Drivers'])

df['Restricted No. of Drivers'] = labelEncoder.transform(df['Restricted No. of Drivers'])


# In[57]:


df.info()


# In[58]:


sns.pairplot(df,hue='Age')


# In[69]:


X  = df.iloc[:,1:-1]
Y =  df.iloc[:,10]
X = X.values
Y = Y.values


# In[70]:


imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X)
X= imputer.transform(X)


# In[71]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)


# In[78]:


linear_regression = LinearRegression()

linear_regression.fit(X_train,Y_train)

linear_regression.score(X_train, Y_train)


# In[79]:


prediction = linear_regression.predict(X_test)


# In[80]:


prediction


# In[ ]:




