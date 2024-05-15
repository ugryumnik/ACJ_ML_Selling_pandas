#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


lightgbm_preds = pd.read_csv("submissions/lightgbm.csv")
rnn_preds = pd.read_csv("submissions/rnn.csv")

preds = [lightgbm_preds, rnn_preds]
weights = [1, 0.6]

submit = lightgbm_preds.copy()

submit['score'] = sum(pred['score'] * weights[i] for i, pred in enumerate(preds)) / sum(weights)

submit.to_csv('submissions/blended_lgbm_rnn.csv', index=False)


# In[3]:





