#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import mode


# In[2]:


# Считываем данные
df = pd.read_csv('data/dataset.csv')
sample_submission = pd.read_csv('data/submission_example.csv')


# In[3]:


# Переименовываем названия стобцов

df.rename(columns={'clientbankpartner_pin': 'partner',
                   'client_pin': 'client',
                   'partner_src_type_ccode': 'type',
                   'client_start_date': 'time',
                   'partnerrolestart_date': 'start_time'}, inplace=True)

df


# ### Решим проблему различных типов у партнера

# In[4]:


nunique_types = df.groupby('partner')['type'].nunique()
partners_multiple_types = nunique_types[nunique_types>1].index


# In[5]:


def get_mode(x):
    return mode(x, keepdims=True)[0][0]

freq_types = df[df.partner.isin(partners_multiple_types)].groupby('partner')['type'].agg(get_mode).to_dict()


# In[6]:


df.loc[df.partner.isin(partners_multiple_types), 'type'] = \
            df.loc[df.partner.isin(partners_multiple_types), 'partner'].map(freq_types)


# In[7]:


assert all(df.groupby('partner')['type'].nunique() == 1)


# In[ ]:





# ### Удалим редкие типы

# In[8]:


df['type'].value_counts()


# In[9]:


df.loc[df.type.isin([0, 2]), 'type'] = 4 # Очень редкие типы, заменим их на самый частый


# In[10]:


df['type'] -= 1 # Нулевой тип удалили, вычтем единицу, чтобы они с нуля начинались


# In[ ]:





# In[11]:


df.to_csv('data/dataset_fixed.csv', index=False)


# In[ ]:





# In[ ]:





# В данном соревновании это не требовалось, но в будущем может стать слишком много типов партнеров, поэтому следует убедиться, что они все лежат в промежутке от 0 до n-1. Это в особенности пригодится при построении нейронной сети

# In[12]:


# unique_types = np.unique(df['type'])
# map_unique_types = dict(zip(unique_types, list(range(len(unique_types)))))
# # map_unique_types_reverse = dict(zip(list(range(len(unique_types))), unique_types))
# map_unique_types


# In[13]:


# df['type'] = df['type'].map(map_unique_types)

