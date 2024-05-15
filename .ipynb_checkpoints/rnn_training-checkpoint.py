#!/usr/bin/env python
# coding: utf-8

# # Requirements

# In[1]:


import os
import pickle
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")


print(torch.cuda.is_available())

from rnn_utils import seed_everything
from data_generators import batches_generator
from pytorch_training import train_epoch, validate, inference
    
get_ipython().system('mkdir submissions')
get_ipython().system('mkdir weights')


# In[2]:


with open('constants/uniques.pkl', 'rb') as f:
     uniques = pickle.load(f)
        
with open('constants/di_features.pkl', 'rb') as f:
     di_features = pickle.load(f)    


# In[3]:


# Пути к бакетам

path_to_dataset = 'buckets/train'
dir_with_datasets = os.listdir(path_to_dataset)
dataset_train = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets if x.endswith('pkl')])
dataset_train = dataset_train
print(len(dataset_train))

path_to_dataset = 'buckets/train_full'
dir_with_datasets = os.listdir(path_to_dataset)
dataset_train_full = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets if x.endswith('pkl')])
print(len(dataset_train_full))

path_to_dataset = 'buckets/val'
dir_with_datasets = os.listdir(path_to_dataset)
dataset_val = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets if x.endswith('pkl')])
print(len(dataset_val))

path_to_dataset = 'buckets/test'
dir_with_datasets = os.listdir(path_to_dataset)
dataset_test = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets if x.endswith('pkl')])
print(len(dataset_test))

dataset_train_full[0], dataset_train_full[-1]


# In[4]:


def get_dataloader(type_of_loader, batch_size):
    """
    Возвращается нужный DataLoader
    :param type_of_loader: str (train/train_full/val/test,  Тип датасета)
    :param batch_size: Размер батча
    """
    if type_of_loader == 'train':
        return batches_generator(dataset_train,
                                         batch_size = batch_size,
                                         has_target=True,
                                         shuffle=True,
                                         di_features = di_features
                                        )
    
    elif type_of_loader == 'train_full':
        return batches_generator(dataset_train_full,
                                         batch_size = batch_size,
                                         has_target=True,
                                         shuffle=True,
                                         di_features = di_features
                                        )
    
    elif type_of_loader == 'val':
        return batches_generator(dataset_val,
                                     batch_size = batch_size,
                                     has_target=True,
                                     shuffle=False,
                                     di_features = di_features
                                    )
    
    elif type_of_loader == 'test':
        return batches_generator(dataset_test,
                                     batch_size = batch_size,
                                     has_target=False,
                                     shuffle=False,
                                     di_features = di_features
                                )
                                 
    else:
        print('No such type of dataloader, available are: train, val, test, train_full')


# In[5]:


batch_size = 2**6

dataloader_train = get_dataloader('train',
                                     batch_size = batch_size,
                                    )

c = 0
for batch in dataloader_train:
    c += 1
print('Всего батчей:', c)


# # Model

# In[6]:


# Рассчитываем размерность эмбеддинга

def compute_embed_dim(n_cat: int) -> int:
    return min(600, 2 * round(1.6 * n_cat**0.56))

embedding_projections = dict()
for feat, uniq in uniques.items():
    embedding_projections[feat]=(max(uniq)+1, compute_embed_dim(max(uniq)+1)) # Число уникальных значений и размер эмбеддинга
embedding_projections

with open('constants/embedding_projections.pkl', 'wb') as f:
    pickle.dump(embedding_projections, f)


# In[7]:


from rnn_model import ChurnPredictor # Импортируем класс модели


# In[8]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class cfg:
    use_embeds_pooling = False # Брать ли пулинги от эмбеддингов
    is_bidirectional = True # Двунаправленная ли LSTM
    spatial_dropout = 0.05 # Spatial dropout для эмбеддингов
    rnn_units = 64 # Размер скрытого состояния LSTM
    num_epochs = 10 # Число эпох
    max_lr = 3e-4 # Максимальный learning rate в цикле
    min_lr = max_lr / 5 # Минимальный learning rate в цикле
    early_stopping_epochs = 10 # Число эпох для early stopping
    device = device
    di_features = di_features


seed_everything(0)
model = ChurnPredictor(di_features, embedding_projections, rnn_units=cfg.rnn_units, spatial_dropout=cfg.spatial_dropout, 
            use_embeds_pooling=cfg.use_embeds_pooling, is_bidirectional=cfg.is_bidirectional).to(device)

optimizer = torch.optim.AdamW(lr=cfg.max_lr, params=model.parameters()) # Optimizer
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.min_lr, max_lr=cfg.max_lr, cycle_momentum = False) # Cyclic Scheduler


# # Training

# In[9]:


# model.train()
# best_auc = -1
# best_epoch = -1
# for epoch in range(cfg.num_epochs):
    
#     dataloader_train = get_dataloader('train', batch_size)
    
#     dataloader_val = get_dataloader('val', batch_size)
    
#     train_epoch(model, dataloader_train, cfg, optimizer, scheduler, epoch)
    
#     auc = validate(model, dataloader_val, cfg)
    
#     if auc > best_auc:
#         best_epoch = epoch
#         best_auc = auc
        
#     elif epoch >= best_epoch + cfg.early_stopping_epochs:
#         print('Результат на валидации не улучшался в течение {} эпох'.format(cfg.early_stopping_epochs))
#         break

# print("BEST_AUC", best_auc, "Epoch:", best_epoch)




best_epoch = 9


# ### Training on train_full (without val)

# In[11]:


seed_everything(0)
model = ChurnPredictor(di_features, embedding_projections, rnn_units=cfg.rnn_units, spatial_dropout=cfg.spatial_dropout, 
            use_embeds_pooling=cfg.use_embeds_pooling, is_bidirectional=cfg.is_bidirectional).to(device)

optimizer = torch.optim.AdamW(lr=cfg.max_lr, params=model.parameters())
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.min_lr, max_lr=cfg.max_lr, cycle_momentum = False)


model.train()

print('Результат на валидации лишь отражает точность на последних 3-х месяцах трейна!')

for epoch in range(best_epoch + 1):
    
    dataloader_train = get_dataloader('train_full', batch_size)
    dataloader_val = get_dataloader('val', batch_size) # Эта валидация содержится в трейне!!!
    
    train_epoch(model, dataloader_train, cfg, optimizer, scheduler, epoch)
    
    auc = validate(model, dataloader_val, cfg)


# In[12]:


dataloader_test = get_dataloader('test', batch_size)
inference(model, dataloader_test, device, di_features, path_to_sample_submission='data/submission_example.csv', path_to_save='submissions/rnn.csv')


# In[13]:


torch.save(model, 'weights/rnn.pt') # Сохраним веса


# In[ ]:




