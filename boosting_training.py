#!/usr/bin/env python
# coding: utf-8

# # Requirements

# In[1]:


import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from lightgbm import LGBMClassifier, train, Dataset, Booster
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier


import shap

import optuna

from sklearn.metrics import roc_auc_score

from optuna_tuning import tune_hyperparams # Кастомная функция для тюнинга параметров

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


get_ipython().system('mkdir val_preds')
get_ipython().system('mkdir submissions')
get_ipython().system('mkdir weights')


# In[2]:


# Считываем данные
X_train = pd.read_parquet('data/X_train_processed.parquet').drop(columns=['partition'])
X_train_full = pd.read_parquet('data/X_train_full_processed.parquet').drop(columns=['partition'])
X_val = pd.read_parquet('data/X_val_processed.parquet')
X_test = pd.read_parquet('data/X_test_processed.parquet')
y_train = pd.read_parquet('data/y_train_processed.parquet')
y_train_full = pd.read_parquet('data/y_train_full_processed.parquet')
y_val = pd.read_parquet('data/y_val_processed.parquet')
sample_submission = pd.read_csv('data/submission_example.csv')
submit = sample_submission.copy()


# In[3]:


X_train


# In[4]:


cat_cols = ['type']
num_cols = [col for col in X_train if col not in cat_cols]


# In[5]:


X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

# Нормируем числовые переменные: приводим данные в валидации к масштабу трейна, данные в тесте к масштабу полного трейна
X_val_scaled[num_cols] = X_val[num_cols] * X_train[num_cols].mean() / X_val[num_cols].mean()
X_test_scaled[num_cols] = X_test_scaled[num_cols] * X_train_full[num_cols].mean() / X_test[num_cols].mean()


# # Optuna

# ## Catboost

# In[6]:


# %%time

# params, auc_c = tune_hyperparams(X_train, X_val_scaled, y_train, y_val,
#       model='catboost', n_trials=50)


# In[7]:


# %%time

# model_c = CatBoostClassifier(**params)
# model_c.fit(X_train, y_train, eval_set = (X_val_scaled, y_val), verbose=100)

# # model_c.load_model(f"checkpoints_boostings/catboost_{auc_c:.4f}.bin")
# iters_c = model_c.best_iteration_ + 1

# val_preds = model_c.predict_proba(X_val_scaled)[:, 1]

# auc_c = roc_auc_score(y_val, val_preds)
# print(f"Best iterations is {iters_c}")
# print("\n Validation AUC: %.4f" % auc_c)

# pd.Series(val_preds).to_csv(f'val_preds/catboost_{auc_c:.4f}.csv', index=False)


# In[8]:


# %%time

# params['n_estimators'] = iters_c
# params['use_best_model'] = False

# model_c = CatBoostClassifier(**params)
# model_c.fit(X_train_full, y_train_full, verbose=100)

# test_preds = model_c.predict_proba(X_test_scaled)[:, 1]

# submit['score'] = test_preds

# submit.to_csv(f'submissions/catboost_{auc_c:.4f}.csv', index=False)


# # XGBoost

# In[9]:


# %%time

# params, auc_x = tune_hyperparams(X_train, X_val_scaled, y_train, y_val,
#       model='xgboost', n_trials=50)


# In[10]:


# %%time

# model_x = XGBClassifier(**params)

# model_x.fit(X_train, y_train, eval_set=[(X_val_scaled, y_val)])
# val_preds = model_x.predict_proba(X_val_scaled.values)[:, 1]
# iters_x = model_x.best_iteration + 1

# auc_x = roc_auc_score(y_val, val_preds)
# print(f"Best iterations is {iters_x}")
# print("\n Validation AUC: %.4f" % auc_x)

# pd.Series(val_preds).to_csv(f'val_preds/xgboost_{auc_x:.4f}.csv', index=False)


# In[11]:


# %%time

# # params['n_estimators'] = 10

# params_copy = params.copy()
# params_copy['n_estimators'] = iters_x
# del params_copy['early_stopping_rounds']

# model_x = XGBClassifier(**params_copy)

# model_x.fit(X_train_full, y_train_full, verbose=100)
# test_preds = model_x.predict_proba(X_test_scaled.values)[:, 1]


# submit['score'] = test_preds
# submit.to_csv(f'submissions/xgboost_{auc_x:.4f}.csv', index=False)


# # LightGBM

# In[12]:


# %%time

# # Тюнинг параметров

# params, auc_l = tune_hyperparams(X_train, X_val_scaled, y_train, y_val,
#       model='lightgbm', n_trials=50)


# In[13]:


# Уже определены оптимальные параметры

params = {
    'subsample': 0.8865310038369543,
    'learning_rate': 0.13475353156897893,
    'colsample_bytree': 0.7228531773341953,
    'reg_alpha': 0.893320872777366,
    'reg_lambda': 0.6477119263843745,
    'num_leaves': 49,
    'max_depth': 6,
    'min_child_samples': 107,
    'objective': 'binary',
    'n_estimators': 1000,
    'early_stopping_round': 50,
    'random_state': 42,
    'verbosity': -1,
}


# In[14]:


# Датасеты для LightGBM

train_l = Dataset(X_train, y_train
)

train_full_l = Dataset(X_train_full, y_train_full
)

val_l = Dataset(X_val_scaled, y_val
)


# In[15]:


model_l = train(
    params,
    train_set=train_l,
    valid_sets=(val_l),
)
iters_l = model_l.best_iteration + 1

val_preds = model_l.predict(X_val_scaled)

auc_l = roc_auc_score(y_val, val_preds)
print(f"Best iterations is {iters_l}")
print("\n Validation AUC: %.4f" % auc_l)

# Во время экспериментов лучше сохранять в названии файла точность на валидации
# pd.Series(val_preds).to_csv(f'val_preds/lightgbm_{auc_l:.4f}.csv', index=False)
pd.Series(val_preds).to_csv(f'val_preds/lightgbm.csv', index=False)


# In[16]:


model_l.save_model('weights/lightgbm.txt', importance_type='gain')


# In[17]:


# Инференс

params_copy = params.copy()
params_copy['n_estimators'] = iters_l
del params_copy['early_stopping_round']

model_l = train(
    params_copy,
    train_set=train_full_l,
)
iters_l = model_l.best_iteration + 1

test_preds = model_l.predict(X_test_scaled)

submit['score'] = test_preds
# Во время экспериментов лучше сохранять в названии файла точность на валидации
# submit.to_csv(f'submissions/lightgbm_{auc_l:.4f}.csv', index=False)
submit.to_csv(f'submissions/lightgbm.csv', index=False)


# In[18]:


model_l.save_model('weights/lightgbm_without_validation.txt', importance_type='gain')


# # Интерпретация модели

# In[19]:


# Будем использовать модель, обученную с валидацией
model_l = Booster(model_file='weights/lightgbm.txt')
model_l.params['objective'] = 'binary'