#!/usr/bin/env python
# coding: utf-8

# Готовим датасет для RNN

# # 1. Requirements

# In[1]:


# Библиотеки
import warnings
warnings.filterwarnings("ignore")
from pandas.tseries.offsets import DateOffset

import pandas as pd
import numpy as np
import sklearn

from collections import defaultdict
import pickle
from rnn_utils import read_parquet_dataset_from_local, create_padded_buckets, transform_actions_to_sequences, seed_everything
from tqdm import tqdm
import os


# In[2]:


get_ipython().system('mkdir constants')

get_ipython().system('mkdir buckets')

get_ipython().system('mkdir partitions')
get_ipython().system('mkdir partitions/train')
get_ipython().system('mkdir partitions/val')
get_ipython().system('mkdir partitions/test')
get_ipython().system('mkdir partitions/train_full')


# In[3]:


# Считываем данные
df = pd.read_csv('data/dataset_fixed.csv')
sample_submission = pd.read_csv('data/submission_example.csv')


# In[4]:


# Необходимые пороги
start_train_threshold = '2020-03-01' # Должно быть хотя бы одно привлечение позже этой даты, иначе партнер уже "ушел"
train_threshold = '2020-06-01' # Конец трейна (привлечения позже этой даты не подаем в X_train)
val_threshold = '2020-09-01' # Конец валидации (привлечения позже этой даты не подаем в X_val)
test_threshold = '2020-12-01' # Конец теста (привлечения позже этой даты не подаем в X_test)


# # Base Processing

# In[5]:


# object -> datetime
df['time'] = pd.to_datetime(df['time'])
df['start_time'] = pd.to_datetime(df['start_time'])
df


# In[6]:


# Формируем фичи по датам

df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month - 1
df['day_of_week'] = df['time'].dt.day_of_week


df['year_start'] = df['start_time'].dt.year
df['month_start'] = df['start_time'].dt.month - 1
df['day_of_week_start'] = df['start_time'].dt.day_of_week

df['time_col'] = df['time'] # Сохраняем оригинальное время (пригодится для сортировки)


# In[7]:


# Нарезаем датасет на train/train_full/val/test
# Train full - полная выборка, нужна для обучения без валидации (это осуществяется непосредственно перед инференсом)

X_train = df[df.time < train_threshold] # 2019-03-01 - 2020-03-01
X_val = df[(df.time.min() + DateOffset(months=3) < df.time) & (df.time < val_threshold)] # 2019-06-01 - 2020-06-01

X_train_full = df[df.time < val_threshold] # 2019-03-01 - 2020-09-01

X_test = df[df.time < test_threshold] # 2019-03-01 - 2020-12-01
X_test = X_test[X_test.time > df.time.min() + DateOffset(months=3)] # 2019-06-01 - 2020-12-01


# In[8]:


def get_diffs(x): # Сколько времени с предыдущего привлечения до текущего прошло
    return [-1] + (np.array(x[:-1])-np.array(x[1:])).tolist()


def add_features(data, has_target, last_attraction_threshold=None, 
                 end_threshold=None, target_end_threshold=None, num_partition=None):
    
    """
    Добавляет фичи в данные
    :param data: pd.DataFrame (DataFrame с привлечениями)
    :param has_target: bool (Есть таргет или нет)
    :param last_attraction_threshold: str/pd.DateTime (хоть одно привлечение должно быть позже этой даты,
    иначе считаем партнера уже ушедшим)
    :param end_threshold: str/pd.DateTime (берем привлечения до этой даты)
    :param target_end_threshold: str/pd.DateTime (таргет определяем по периоду от end_threshold до этой даты)
    :param num_partition: int, optional (Номер партиции)
    :return: pd.DataFrame, pd.DataFrame - фичи и таргет (для теста только фичи)
    """
                
    data = data[data.time < end_threshold]
    
    
    # Оставляем только не ушедших партнеров
    max_time_of_attraction = data.groupby('partner').agg({'time': 'max'})
    appropriate_partners = max_time_of_attraction[max_time_of_attraction.time>=last_attraction_threshold].index
    data = data[data.partner.isin(appropriate_partners)]
    
    # Сколько дней осталось до порога (начиная с которого идет таргет)
    data['time_left'] = (pd.to_datetime(end_threshold) - data['time']).astype(int)//10**9 / 3600 // 24
    
    # Переводим в дни
    data['start_time'] = data['start_time'].astype(int)//10**9 / 3600 // 24
    
    if has_target:
        # Если в нужный период есть привлечение, то 0, иначе 1
        y = pd.DataFrame({'partner': np.unique(data.partner), 'score': 0})
        y.loc[y.partner.isin(np.unique(df[(end_threshold <= df.time) & ( df.time < target_end_threshold)]['partner'])), 'score'] = 1
    
    
    data = data.sort_values(['partner', 'time_col', 'client'], ascending=True)
    
    # # Номер привлечения
    # data['num_attraction'] = data.groupby('partner')['time'].transform(lambda x: list(range(1, len(x)+1))).values
    # # Всего привлечений
    # data['count_attractions'] = data.groupby('partner')['num_attraction'].transform('max')
    
    if num_partition is not None:
        # Записываем номер партиции
        data['partition'] = num_partition
        y['partition'] = num_partition
    
    # # Дней с прошлого привлечения
    # data['diff'] = data.groupby('partner')['time_left'].transform(get_diffs).values
    
        
    if has_target:
        return data, y
    else:
        return data


# # Train and Train_full

# Создадим Train и Full train с аугментацией (нарезая датасет на части).
# 
# Пример:
# 
# Партиция 0: данные с 2019-03-01 по 2020-02-28, таргет с 2020-03-01 по 2020-05-31
# 
# Партиция 1: данные с 2019-02-15 по 2020-02-13, таргет с 2020-02-15 по 2020-05-16
# 
# Партиция 2: данные с 2019-02-01 по 2020-01-29, таргет с 2020-02-01 по 2020-05-01
# 
# ...

# In[9]:


delta = DateOffset(days=15) # С каким промежутком нарезать
num_partitions = 26 # Сколько партиций


# In[10]:


X_train_list = []
y_train_list = []


for num_partition in range(num_partitions):
    # Пороги для партиции
    start_train_threshold_new = pd.to_datetime(start_train_threshold) - delta * num_partition
    train_threshold_new = pd.to_datetime(train_threshold) - delta * num_partition
    val_threshold_new = pd.to_datetime(val_threshold) - delta * num_partition
    
    # Данные для одной партиции
    X_train_part, y_train_part = add_features(X_train.copy(), has_target=True, last_attraction_threshold=start_train_threshold_new, 
                                end_threshold=train_threshold_new, target_end_threshold=val_threshold_new,
                               num_partition=num_partition)
    
    X_train_list.append(X_train_part)
    y_train_list.append(y_train_part)
    # print(X_train_part.shape)
    # print(y_train_part.score.value_counts().to_dict())

# Объединяем партиции в единый датасет
X_train = pd.concat(X_train_list, axis=0, ignore_index=True)
y_train = pd.concat(y_train_list, axis=0, ignore_index=True)
X_train.shape, y_train.shape


# Теперь то же самое для full train

# In[11]:


X_train_full_list = []
y_train_full_list = []


for num_partition in range(num_partitions):
    start_train_threshold_new = pd.to_datetime(train_threshold) - delta * num_partition
    train_threshold_new = pd.to_datetime(val_threshold) - delta * num_partition
    val_threshold_new = pd.to_datetime(test_threshold) - delta * num_partition

    
    X_train_part, y_train_part = add_features(X_train_full.copy(), has_target=True, last_attraction_threshold=start_train_threshold_new, 
         end_threshold=train_threshold_new, target_end_threshold=val_threshold_new,
                               num_partition=num_partition)
    
    
    
    X_train_full_list.append(X_train_part)
    y_train_full_list.append(y_train_part)
    # print(X_train_part.shape)
    # print(y_train_part.score.value_counts().to_dict())
    
    
X_train_full = pd.concat(X_train_full_list, axis=0, ignore_index=True)
y_train_full = pd.concat(y_train_full_list, axis=0, ignore_index=True)
X_train_full.shape, y_train_full.shape


# # Val and Test

# Теперь валидация и тест

# In[12]:


X_val, y_val = add_features(X_val.copy(), has_target=True, last_attraction_threshold=train_threshold,
                    end_threshold=val_threshold, target_end_threshold=test_threshold)

X_test = add_features(X_test.copy(), has_target=False, last_attraction_threshold=val_threshold,
                    end_threshold=test_threshold)


# # Feature selection

# Выберем нужные нам фичи

# In[13]:


X_train.columns


# In[14]:


X_train.head(3)


# In[15]:


# Сохраним X_train без аугментации (пригодится для подсчета характеристик, основанных на распределении трейна)

if 'partition' in X_train:
    X_train_base = X_train[X_train.partition == 0].drop(columns=['partition'])
else:
    X_train_base = X_train


# In[16]:


# time_col = 'time_col'
# instance_col = 'partner'

# num_cols = ['time', 'diff', 'num_attraction']
# cat_cols = ['month', 'day_of_week']

# fixed_num_cols = ['start_time', 'count_attractions']
# fixed_cat_cols = ['type', 'month_start', 'day_of_week_start']

# target_col = 'score'


time_col = 'time_col' # Время привлечения (технический столбец)
instance_col = 'partner' # Столбец, опредедяющий одну сущность в данных (технический столбец)

num_cols = ['time_left'] # Числовые переменные (например, сколько дней осталось до начала таргета)
cat_cols = [] # Категориальные переменные (например, месяц привлечения)

fixed_num_cols = [] # Фиксированные числовые переменные (например, время начала работы партнера)
fixed_cat_cols = ['type'] #  категориальные переменные (например, тип партнера, он не зависит от времени)

target_col = 'score' # Целевая переменная

fixed_cols = fixed_num_cols + fixed_cat_cols # Все фиксированные столбцы
non_fixed_cols = num_cols + cat_cols # Все нефиксированные столбцы
dense_cols = num_cols + fixed_num_cols # Все числовые столбцы
non_dense_cols = cat_cols + fixed_cat_cols # Все категориальные столбцы

has_num = len(num_cols) != 0 # Есть ли в данных числовые переменные 
has_cat = len(cat_cols) != 0 # Есть ли в данных категориальные переменные 
has_fixed_num = len(fixed_num_cols) != 0 # Есть ли в данных фиксированные числовые переменные 
has_fixed_cat = len(fixed_cat_cols) != 0 # Есть ли в данных фиксированные категориальные переменные 


# In[17]:


di_features = {} # Словарь с переменными

di_features['cat_cols'] = cat_cols
di_features['num_cols'] = num_cols
di_features['time_col'] = time_col
di_features['instance_col'] = instance_col
di_features['target_col'] = target_col
di_features['fixed_num_cols'] = fixed_num_cols
di_features['fixed_cat_cols'] = fixed_cat_cols
di_features['has_num'] = has_num
di_features['has_cat'] = has_cat
di_features['has_fixed_num'] = has_fixed_num
di_features['has_fixed_cat'] = has_fixed_cat
di_features['dense_cols'] = dense_cols
di_features['non_dense_cols'] = non_dense_cols



with open('constants/di_features.pkl', 'wb') as f:
    pickle.dump(di_features, f)


# # RNN processing

# Подготовка датасета для подачи в RNN

# In[18]:


class cfg:
    number_of_bins_for_num_cols = 12 # На сколько бинов разбивать числовые переменные
    max_length = 50 # Максимальная длина (если больше, берем последние 50)
    num_partitions_train = num_partitions # Сколько партиций в трейне
    num_partitions_val = 1 # Сколько партиций в валидации
    num_partitions_test = 1 # Сколько партиций в тесте


# In[19]:


# Нарезаем числовые фичи на равные по численности промежутки (точнее, пока сохраняем границы)

num_bins = cfg.number_of_bins_for_num_cols

dense_features_buckets = dict()

for feat in dense_cols:
    if feat in num_cols:
        dense_features_buckets[feat]=pd.qcut(X_train_base[feat], num_bins, labels=False, retbins=True, duplicates='drop')[1]
    elif feat in fixed_num_cols:
        dense_features_buckets[feat]=pd.qcut(X_train_base.drop_duplicates(subset=[instance_col])[feat], 
                                             num_bins, labels=False, retbins=True, duplicates='drop')[1]

with open('constants/dense_features_buckets.pkl', 'wb') as f:
    pickle.dump(dense_features_buckets, f)


# In[20]:


# Встречаемость числа привлечений (например, у 767 партнеров по одному привлечению)
lens_count=X_train_base[instance_col].value_counts().value_counts().sort_index()
lens_count


# In[21]:


# Мало у кого больше 50 привлечений
lens_count.loc[lens_count.index > cfg.max_length].sum()


# In[22]:


max_length = cfg.max_length

lens=X_train_base[instance_col].value_counts()

# Разобьем длины на 25 частей. Для каждой части выбирается максимальная длина M. Тогда любая последовательность из этой
# будет подвергаться паддингу до длина M.
pad_borders=pd.qcut(lens, 25, labels=False, retbins=True, duplicates='drop')[1]
pad_borders=np.append(pad_borders, max_length)
pad_borders.sort()
pad_borders = pad_borders[pad_borders<=max_length]
pad_borders=pad_borders.astype('int')
pad_borders


# In[23]:


keys=list(range(1, lens_count[lens_count.index<=max_length].index[-1]+1))
values=[1]
for i in range(1, len(pad_borders)):
    values+=[pad_borders[i]]*(pad_borders[i]-pad_borders[i-1])
length_to_pad=dict(zip(keys, values)) # Длина -> Длина с паддингом
with open('constants/length_to_pad.pkl', 'wb') as f:
    pickle.dump(length_to_pad, f)
# length_to_pad


# # Partitions

# In[24]:


def write_partitions(df, path, num_partitions):
    """
    Создает и сохраняет партиции для валидации и теста (необходимо для больших датасетов)
    :param df: pd.DataFrame (Данные)
    :param path: str (Куда сохранять)
    :param num_partitions: int (Сколько партиций делать)
    :return: None
    """

    ids=sorted(df[instance_col].unique()) # Все ID
    count_ids=len(ids)

    for i in tqdm(range(num_partitions)):
        index_to_write=ids[int(count_ids*i/num_partitions):int(count_ids*(i+1)/num_partitions)] # Берем 1/num_partitions долю от всех ID
        data=df[df[instance_col].isin(index_to_write)]
        num=str(i)
        if len(num)==1:
            num='0'+num
        data.to_parquet(f'{path}/part_{num}')
        
        
        
def write_partitions_train(df, path, num_partitions):
    """
    Сохраняет партиции для train и train_full (для них уже готовы партиции)
    :param df: pd.DataFrame (Данные)
    :param path: str (Куда сохранять)
    :param num_partitions: int (Сколько партиций делать)
    :return: None
    """
    for i in range(num_partitions):
        data = df[df.partition == i]
        num=str(i)
        if len(num)==1:
            num='0'+num
        data.to_parquet(f'{path}/part_{num}')


# In[25]:


write_partitions_train(X_train, 'partitions/train', num_partitions=cfg.num_partitions_train)
write_partitions_train(X_train_full, 'partitions/train_full', num_partitions=cfg.num_partitions_train)
write_partitions(X_val, 'partitions/val', num_partitions=cfg.num_partitions_val)
write_partitions(X_test, 'partitions/test', num_partitions=cfg.num_partitions_test)


# # Uniques

# In[26]:


# Уникальные значения фичей при подаче в нейронную сеть

uniques = defaultdict(set)

for feat in tqdm(fixed_cat_cols + cat_cols):
    uniques[feat] = X_train_base[feat].unique()
    
for feat in num_cols + fixed_num_cols:
    uniques[feat]=set(range(0, len(dense_features_buckets[feat])-1)) # Число границ - 1

    
with open('constants/uniques.pkl', 'wb') as f:
     pickle.dump(uniques, f)


# ## Buckets

# Создаем бакеты, которые будем подгружать при обучении нейронной сети

# In[27]:


# with open('constants/length_to_pad.pkl', 'rb') as f:
#     length_to_pad = pickle.load(f)
    
# with open('constants/dense_features_buckets.pkl', 'rb') as f:
#     dense_features_buckets = pickle.load(f)


# In[28]:


def create_buckets_from_actions(path_to_dataset, save_to_path, frame_with_ids = None, 
                                     num_parts_to_preprocess_at_once: int = 1, 
                                     num_parts_total=10, has_target=False):
    
    
    """
    Преобразует датасет в бакеты, готовые для подачи в даталоадер
    Читает num_parts_to_preprocess_at_once частей датасета в память
    Преобразует вещественные и численные признаки к категориальным (используя np.digitize и подготовленные бины)
    Формирует фрейм с транзакциями в виде последовательностей с помощью transform_actions_to_sequences.
    Реализует технику sequence_bucketing и сохраняет словарь обработанных последовательностей в .pkl файл
    
    :param path_to_dataset: str (Где партиции сохранены)
    :param save_to_path: str (Куда сохранять бакеты)
    :param frame_with_ids: pd.DataFrame, optional (DataFrame с индексами. По сути, к нему мерджим наши последовательные данные)
    :param num_parts_total: int (Всего партиций)
    :param has_target: bool (Есть таргет или нет)
    :return: None
    
    """
    
    block = 0
    for step in range(0, num_parts_total, num_parts_to_preprocess_at_once):
        actions_frame = read_parquet_dataset_from_local(path_to_dataset, step, num_parts_to_preprocess_at_once, 
                                                             verbose=False)
        
        
        for dense_col in dense_cols:
            # Разбиваем числовые фичи на бины
            actions_frame[dense_col] = np.digitize(actions_frame[dense_col], bins=dense_features_buckets[dense_col])
            # Правим бин, если выходим за границы:
            actions_frame[dense_col] = actions_frame[dense_col].apply(lambda x: max(1, x))
            actions_frame[dense_col] = actions_frame[dense_col].apply(lambda x: min(len(dense_features_buckets[dense_col])-1, x)) - 1
            
        
        # Переводимм датасет в последовательные данные (для каждого партнера имеем последовательность привлечений)
        seq = transform_actions_to_sequences(actions_frame, num_last_actions=max_length,
                                                 di_features=di_features)
        
        
        if 'partition' in actions_frame:
            assert actions_frame['partition'].nunique() == 1
            partition = actions_frame['partition'].iloc[0]
        
        if len(num_cols):
            seq['sequence_length'] = seq['num_cols'].apply(lambda x: len(x[0])) # Длина последовательности
        else:
            seq['sequence_length'] = seq['cat_cols'].apply(lambda x: len(x[0]))
        

        if frame_with_ids is not None:
            if 'partition' in actions_frame:
                seq = seq.merge(frame_with_ids[frame_with_ids.partition == partition], on='partner')
            else:
                seq = seq.merge(frame_with_ids, on='partner')

        block_as_str = str(block)
        if len(block_as_str) == 1:
            block_as_str = '00' + block_as_str
        else:
            block_as_str = '0' + block_as_str
            
            
        # Наконец, создаем бакеты
        processed_fragment =  create_padded_buckets(seq, length_to_pad, has_target=has_target, 
                                                    save_to_file_path=os.path.join(save_to_path, 
                                                                                   f'processed_chunk_{block_as_str}.pkl'),
                                                   di_features=di_features)
        block += 1


# In[29]:


get_ipython().system("rm -r 'buckets/train'")
get_ipython().system("mkdir 'buckets/train'")

create_buckets_from_actions('partitions/train', 
                                save_to_path='buckets/train',
                                frame_with_ids=y_train, num_parts_to_preprocess_at_once=1, 
                                 num_parts_total=cfg.num_partitions_train, has_target=True)
print('BUCKETS CREATED!')


# In[30]:


get_ipython().system("rm -r 'buckets/train_full'")
get_ipython().system("mkdir 'buckets/train_full'")

create_buckets_from_actions('partitions/train_full', 
                                save_to_path='buckets/train_full',
                                frame_with_ids=y_train_full, num_parts_to_preprocess_at_once=1, 
                                 num_parts_total=cfg.num_partitions_train, has_target=True)
print('BUCKETS CREATED!')


# In[31]:


get_ipython().system("rm -r 'buckets/val'")
get_ipython().system("mkdir 'buckets/val'")

create_buckets_from_actions('partitions/val', 
                                save_to_path='buckets/val',
                                frame_with_ids=y_val, num_parts_to_preprocess_at_once=1, 
                                 num_parts_total=cfg.num_partitions_val, has_target=True)
print('BUCKETS CREATED!')


# In[32]:


get_ipython().system("rm -r 'buckets/test'")
get_ipython().system("mkdir 'buckets/test'")

create_buckets_from_actions('partitions/test', 
                                save_to_path='buckets/test',
                                frame_with_ids=None, num_parts_to_preprocess_at_once=1, num_parts_total=cfg.num_partitions_test, has_target=False)
print('BUCKETS CREATED!')

