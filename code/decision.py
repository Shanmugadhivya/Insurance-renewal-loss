#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries
# 

# In[3]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing the Dataset
# 

# In[4]:


df = pd.read_csv('G:/losses.csv')


# ## Find the number of rows and columns

# In[5]:


df.shape  


# In[6]:


df.head()  


# ## Preparing the Data
# 

# In[7]:


X = df.drop('smoker', axis=1)  
y = df['smoker']  


# ## Divide the data into train and test sets
# 

# In[8]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  


# ## Training and Making Predictions
# 

# In[9]:


from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train)  


# ## predictions on the test data

# In[10]:


y_pred = classifier.predict(X_test)  


# In[11]:


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

