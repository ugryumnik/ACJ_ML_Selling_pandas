#!/usr/bin/env python
# coding: utf-8

# # Reading data

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")
from pandas.tseries.offsets import DateOffset
from collections import Counter


# In[2]:


df = pd.read_csv('data/dataset_fixed.csv')
sample_submission = pd.read_csv('data/submission_example.csv')


# In[3]:


# Необходимые пороги
start_train_threshold = '2020-03-01' # Должно быть хотя бы одно привлечение позже этой даты, иначе партнер уже "ушел"
train_threshold = '2020-06-01' # Конец трейна (привлечения позже этой даты не подаем в X_train)
val_threshold = '2020-09-01' # Конец валидации (привлечения позже этой даты не подаем в X_val)
test_threshold = '2020-12-01' # Конец теста (привлечения позже этой даты не подаем в X_test)


# In[4]:


df['time'] = pd.to_datetime(df['time'])
df['start_time'] = pd.to_datetime(df['start_time'])
df


# In[5]:


# Нарезаем датасет на train/train_full/val/test
# Train full - полная выборка, нужна для обучения без валидации (это осуществяется непосредственно перед инференсом)

X_train = df[df.time < train_threshold] # 2019-03-01 - 2020-03-01
X_val = df[(df.time.min() + DateOffset(months=3) < df.time) & (df.time < val_threshold)] # 2019-06-01 - 2020-06-01

X_train_full = df[df.time < val_threshold] # 2019-03-01 - 2020-09-01

X_test = df[df.time < test_threshold] # 2019-03-01 - 2020-12-01
X_test = X_test[X_test.time > df.time.min() + DateOffset(months=3)] # 2019-06-01 - 2020-12-01


# # Feature engineering

# In[6]:


# Сколько партнеров пришли в этот день
partners_by_start_time = X_train.drop_duplicates('partner')['start_time'].value_counts()


# In[7]:


def num_unique_values(x):
    # Число уникальных значений фичи для партнера
    return len(set(x))

def freq(x):
    # Самое частое значение фичи для партнера
    return Counter(x).most_common(1)[0][0]


def get_fixed(x):
    # Получаем фиксированное значение (например, тип партнера не зависит от времени, тут для партнера мы получим одно число - его тип)
    return x.iloc[0]


def get_last_n_and_pad(x, n):
    # Получаем время до порога для n последних привлечений. Делаем паддинг (если меньше 5 привлечений у партнера)
    x = x[-n:]

    if isinstance(x, int):
        x = [x]
    else:
        x = x.tolist()

    return (n - len(x)) * [-1] + x


def data_to_last_actions(data):
    # Получаем время до порога для n последних привлечений в нужном формате.
    
    n = 5
    group_time = data.groupby('partner')['time_left'].agg(lambda x: get_last_n_and_pad(x, n))
    group_time = pd.DataFrame(group_time)
    group_time.columns=['times']
    group_time[[f'time_left_{i}_attraction' for i in range(n)]] = group_time['times'].apply(lambda x: pd.Series(x)).values
    group_time.drop(columns=['times'], inplace=True)


    data = data.drop_duplicates(subset=['partner'])
    data = data.merge(group_time, left_on = 'partner', right_index=True).drop(columns=['time_left'])
    return data

def get_diffs(x):
    # Сколько времени с предыдущего привлечения до текущего прошло
    x = x.values.tolist()
    x = [np.nan] + (np.array(sorted(x, reverse=True)[:-1]) - np.array(sorted(x, reverse=True)[1:])).tolist()

    return x
    
def get_diff_more_than_90_count(x):
    # Сколько у партнера было 'перерывов' в привлечениях дольше 90 дней
    return len(x[x>90])


def get_features(data, target, last_attraction_threshold, threshold):
    """
    Добавляет фичи в данные, аггрерирует их
    :param data: pd.DataFrame (DataFrame с привлечениями)
    :param target: pd.DataFrame (DataFrame с таргетом, для тестовой выборки None)
    :param last_attraction_threshold: str/pd.DateTime (хоть одно привлечение должно быть позже этой даты,
    иначе считаем партнера уже ушедшим)
    :param threshold: str/pd.DateTime (берем привлечения до этой даты)
    :return: pd.DataFrame, pd.DataFrame - фичи и таргет (для теста только фичи)
    """
    
    
    aggs = {
        "start_time": get_fixed,
        "type": [get_fixed],
        "time_left": ['mean', 'median', 'sum', 'std', 'min', 'max'],
        "time_days": ['mean', 'median', 'min', 'max'],
        'start_time': get_fixed,
        'day_of_week': [freq, num_unique_values],
        'month': [freq, num_unique_values],
        'year': [freq, num_unique_values],
        'diff': ['mean', 'median', 'std', get_diff_more_than_90_count, 'min', 'max', 'sum'] # num_unique_values
        # 'client': ['median', 'min', 'max']
        }
    
    
    # Оставляем только не ушедших партнеров
    max_time_of_attraction = data.groupby('partner').agg({'time': 'max'})
    appropriate_partners = max_time_of_attraction[max_time_of_attraction.time>=last_attraction_threshold].index
    data = data[data.partner.isin(appropriate_partners)]
    
    
    # Time
    data['time'] = pd.to_datetime(data['time'])
    data['time_left'] = (pd.to_datetime(threshold) - data['time']).astype(int)//10**9 / 3600 // 24
    data['time_days'] = data['time'].astype(int)//10**9 / 3600 // 24
    data['day_of_week'] = data['time'].dt.day_of_week
    data['month'] = data['time'].dt.month
    data['year'] = data['time'].dt.year
    
    data.drop(columns=['time'], inplace=True)
    

    
    data.sort_values(['partner', 'time_left'], inplace=True)
    data['diff'] = data.groupby('partner')['time_left'].transform(get_diffs)
    
    
    cols_to_agg = list(aggs.keys())
    
    # Начинаем аггрегацию данных
    data_agg = data[['partner', cols_to_agg[0]]].groupby("partner", as_index=True).agg({cols_to_agg[0]: aggs[cols_to_agg[0]]})
    for feat in cols_to_agg[1:]:
        data_agg = pd.concat([data_agg, data[['partner', feat]].groupby("partner", as_index=True).agg({feat: aggs[feat]})], axis=1)
        
    
    # Исправляем названия столбцов
    data_agg.columns = data_agg.columns.values.tolist()[:1] +  ['_'.join(col).strip('_') for col in data_agg.columns.values[1:]]
    data_agg.columns = [col if 'get_fixed' not in col else col[:-10] for col in data_agg.columns]
    if 'diff_get_diff_more_than_90_count' in data_agg:
        data_agg.rename(columns={'diff_get_diff_more_than_90_count': 'diff_more_than_90'}, inplace=True)
        
    
    data_agg.insert(2, 'count', data.groupby('partner').size()) # Число привлечений
    
    
    # Заполняем наны для партнеров с одним привлечением
    std_cols = [col for col in data_agg.columns if 'std' in col]
    data_agg.loc[:, std_cols] = data_agg.loc[:, std_cols].fillna(-1)
    # data_agg.fillna(0, inplace=True)
    
    
    # Start time
    # data_agg['partners_by_start_time'] = data_agg['start_time'].map(partners_by_start_time.to_dict()).fillna(1)
    data_agg['start_time'] = pd.to_datetime(data_agg['start_time'])
    data_agg['start_time_left'] = (pd.to_datetime(threshold) - data_agg['start_time']).astype(int)//10**9 / 3600 // 24
    data_agg['start_time_days'] = data_agg['start_time'].astype(int)//10**9 / 3600 // 24
    data_agg['day_of_week_start'] =data_agg['start_time'].dt.day_of_week
    data_agg['month_start'] = data_agg['start_time'].dt.month
    data_agg['year_start'] = data_agg['start_time'].dt.year
    
    data_agg['start_time'] = data_agg['start_time'].astype(int)//10**9//3600//24
    # data_agg.drop(columns=['start_time'], inplace=True)
    

    
    
    if target is None:
        data_agg = data_agg.loc[sample_submission['clientbankpartner_pin'].values]
    
    
    
    # last_actions_data = data_to_last_actions(data.copy()[['partner', 'time_left']]).set_index('partner')
    # for col in last_actions_data:
    #     if col not in data_agg:
    #         data_agg[col] = last_actions_data[col].values

    
    
    # Для каждого уникального значения категориальной фичи считаем число привлечений и долю привлечений (например, сколько привлечений в марте было)
    # categorial = ['day_of_week', 'month', 'year']
    # for cat in categorial:
    #     catf = data.groupby(['partner', cat]).size()
    #     catf = catf.unstack(fill_value=0.0)
    #     columns = catf.columns.values
    #     catf.columns = [f"{cat}_{col}_count" for col in columns]
    #     data_agg = data_agg.merge(catf, left_index=True, right_on="partner")

    #     catf = catf.div(catf.sum(axis=1), axis=0)
    #     catf.columns = [f"{cat}_{col}_part" for col in columns]
    #     data_agg = data_agg.merge(catf, left_index=True, right_on="partner")
    #     del catf
        
    
    cat_cols = ['type']
    num_cols = [col for col in data_agg.columns if col not in cat_cols+['score']]
    
    data_agg[cat_cols] = data_agg[cat_cols].astype('int')
    data_agg[num_cols] = data_agg[num_cols].astype('float')
    

    if target is not None:
        data_agg = data_agg.join(target.set_index("partner"), "partner")
        return data_agg.drop(columns=["score"]), data_agg[["score"]]
    else:
        return data_agg
    


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

# In[8]:


delta = DateOffset(days=15) # С каким промежутком нарезать
num_partitions = 26 # Сколько партиций


# In[9]:


X_train_list = []
y_train_list = []


for num_partition in range(num_partitions):
    # Пороги для партиции
    start_train_threshold_new = pd.to_datetime(start_train_threshold) - delta * num_partition
    train_threshold_new = pd.to_datetime(train_threshold) - delta * num_partition
    val_threshold_new = pd.to_datetime(val_threshold) - delta * num_partition
    
    
    # Данные для одной партиции
    data = X_train[X_train.time < train_threshold_new]
    y_train_part = pd.DataFrame({'partner': np.unique(data.partner), 'score': 1})
    y_train_part.loc[y_train_part.partner.isin(np.unique(df[(train_threshold_new <= df.time) & ( df.time < val_threshold_new)]['partner'])), 'score'] = 0

    X_train_part, y_train_part = get_features(data.copy(), y_train_part, start_train_threshold_new, train_threshold_new)
    X_train_part['partition'] = num_partition
    
    
    
    X_train_list.append(X_train_part)
    y_train_list.append(y_train_part)
    # print(X_train_part.shape)
    # print(y_train_part.score.value_counts().to_dict())
    
# Объединяем партиции в единый датасет
X_train = pd.concat(X_train_list, axis=0, ignore_index=True)
y_train = pd.concat(y_train_list, axis=0, ignore_index=True)


# In[10]:


X_train_list = []
y_train_list = []


for num_partition in range(num_partitions):
    # Пороги для партиции
    start_train_threshold_new = pd.to_datetime(train_threshold) - delta * num_partition
    train_threshold_new = pd.to_datetime(val_threshold) - delta * num_partition
    val_threshold_new = pd.to_datetime(test_threshold) - delta * num_partition
    
    
    # Данные для одной партиции
    data = X_train_full[X_train_full.time < train_threshold_new]
    y_train_part = pd.DataFrame({'partner': np.unique(data.partner), 'score': 1})
    y_train_part.loc[y_train_part.partner.isin(np.unique(df[(train_threshold_new <= df.time) & ( df.time < val_threshold_new)]['partner'])), 'score'] = 0
    
    X_train_part, y_train_part = get_features(data.copy(), y_train_part, start_train_threshold_new, train_threshold_new)
    X_train_part['partition'] = num_partition
    
    
    X_train_list.append(X_train_part)
    y_train_list.append(y_train_part)
    # print(X_train_part.shape)
    # print(y_train_part.score.value_counts().to_dict())

# Объединяем партиции в единый датасет
X_train_full = pd.concat(X_train_list, axis=0, ignore_index=True)
y_train_full = pd.concat(y_train_list, axis=0, ignore_index=True)


# # Val and Test

# In[11]:


y_val = pd.DataFrame({'partner': np.unique(X_val.partner), 'score': 1})
y_val.loc[y_val.partner.isin(np.unique(df[df.time >= val_threshold]['partner'])), 'score'] = 0
print(y_val.score.value_counts())
y_val


# In[12]:


X_val, y_val = get_features(X_val, y_val, train_threshold, val_threshold)
X_test = get_features(X_test, None, val_threshold, test_threshold)


# # Saving data

# In[13]:


X_train.shape


# In[14]:


X_train.head(3)


# In[15]:


X_test.head(3)


# In[16]:


X_train.to_parquet('data/X_train_processed.parquet')
X_train_full.to_parquet('data/X_train_full_processed.parquet')
X_val.to_parquet('data/X_val_processed.parquet')
X_test.to_parquet('data/X_test_processed.parquet')
y_train.to_parquet('data/y_train_processed.parquet')
y_train_full.to_parquet('data/y_train_full_processed.parquet')
y_val.to_parquet('data/y_val_processed.parquet')


# In[17]:


X_train.describe()


# In[ ]:




