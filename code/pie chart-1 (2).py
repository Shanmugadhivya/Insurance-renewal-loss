#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries functions
# 

# In[8]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt


# ## Importing Dataset

# In[2]:


df = pd.read_csv('G:/policy.csv')


# In[3]:


print(df.head())
df.info()


# ## Data to plot

# In[4]:


labels= ['Age', 'Nationality', 'Type of Coverage', 'No Claim Availed', 'Years of Insurance undertaking', 'Vehicle Age Years', 
         'Year of Purchase', 'Restricted No.insuranceclaim of Drivers',' Mileage Kms','Vehicle Power in KW','Annual Premium ']

colors=['blue', 'yellow', 'green', 'orange', 'pink', 'purple', 'white', 'red','brown','black','pink']

sizes= [1500, 600, 500, 700, 800, 750, 800, 600, 800, 500, 700]

plt.pie(sizes,labels=labels, colors=colors, startangle=90, autopct='%1.1f%%')

plt.axis('equal')

plt.show()


# In[9]:


df = pd.read_csv('G:/claim.csv')


# In[10]:


print(df.head())
df.info()


# In[12]:


labels= ['age', 'sex', 'children', 'smoker', 'region', 'premium charges', 'insuranceclaim']

colors=['blue', 'yellow', 'green', 'orange', 'pink', 'purple', 'red']

sizes= [1500, 600, 800, 750, 850, 900, 600]

plt.pie(sizes,labels=labels, colors=colors, startangle=90, autopct='%1.1f%%')

plt.axis('equal')

plt.show()


# In[5]:


df = pd.read_csv('G:/theft.csv')


# In[6]:


print(df.head())
df.info()


# In[7]:


labels= ['State', 'Rank', 'Make/Model', 'Model Year', 'Thefts']

colors=['blue', 'yellow', 'green', 'orange', 'pink']

sizes= [1500, 600, 900, 800, 750]

plt.pie(sizes,labels=labels, colors=colors, startangle=90, autopct='%1.1f%%')

plt.axis('equal')

plt.show()


# In[ ]:




