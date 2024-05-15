#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
from lightgbm import Booster

import torch

import os
import pickle

import warnings
warnings.filterwarnings('ignore')


# # Boosting

# In[3]:


X_train_full = pd.read_parquet('data/X_train_full_processed.parquet').drop(columns=['partition'])
X_test = pd.read_parquet('data/X_test_processed.parquet')


# In[4]:


cat_cols = ['type']
num_cols = [col for col in X_train_full if col not in cat_cols]


# In[5]:


X_test_scaled = X_test.copy()

X_test_scaled[num_cols] = X_test_scaled[num_cols] * X_train_full[num_cols].mean() / X_test[num_cols].mean()


# In[6]:


model_l = Booster(model_file='weights/lightgbm_without_validation.txt')


# In[7]:


preds = model_l.predict(X_test_scaled)


# In[8]:


submit = pd.read_csv('data/submission_example.csv')

submit['score'] = preds
submit


# In[9]:


submit.to_csv(f'submissions/lightgbm.csv', index=False)


# # RNN

# In[11]:


from rnn_model import ChurnPredictor
from data_generators import batches_generator
from pytorch_training import inference


# In[18]:


with open('constants/di_features.pkl', 'rb') as f:
     di_features = pickle.load(f) 
        
device = torch.device('cpu')


# In[19]:


model_rnn = torch.load('weights/rnn.pt')


# In[20]:


batch_size = 2**6


# In[21]:


# Путь к бакетам

path_to_dataset = 'buckets/test'
dir_with_datasets = os.listdir(path_to_dataset)
dataset_test = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets if x.endswith('pkl')])


# In[26]:


dataloader_test = batches_generator(dataset_test,
                                     batch_size = batch_size,
                                     has_target=False,
                                     shuffle=False,
                                     di_features = di_features)

inference(model_rnn, dataloader_test, device, di_features, path_to_sample_submission='data/submission_example.csv', path_to_save='submissions/rnn.csv')


# # Blending

# In[27]:


lightgbm_preds = pd.read_csv("submissions/lightgbm.csv")
rnn_preds = pd.read_csv("submissions/rnn.csv")

preds = [lightgbm_preds, rnn_preds]
weights = [1, 0.6]

submit = lightgbm_preds.copy()

submit['score'] = sum(pred['score'] * weights[i] for i, pred in enumerate(preds)) / sum(weights)

submit.to_csv('submissions/blended_lgbm_rnn.csv', index=False)


# In[28]:


submit


# In[ ]:




