from typing import Dict
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
import torch


def seed_everything(seed):
    """
    Обеспечивает воспроизводимость экспериментов
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    


def pad_sequence(array, max_len) -> np.array:
    """
    принимает список списков (array) и делает padding каждого вложенного списка до max_len
    :param array: список списков
    :param max_len: максимальная длина до которой нужно сделать padding
    :return: np.array после padding каждого вложенного списка до одинаковой длины
    """
    add_zeros = max_len - len(array[0])
    return np.array([list(x) + [-1] * add_zeros for x in array])


def truncate(x, num_last_actions=750):
    return x.values.transpose()[:, -num_last_actions:].tolist()
 

def transform_actions_to_sequences(actions_frame: pd.DataFrame,
                                        num_last_actions=750, di_features=None) -> pd.DataFrame:
    """
    принимает frame с привлечениями партнера, берет num_last_actions привлечений,
    возвращает новый pd.DataFrame со следующими столбцами:
    1) instance_col (в данном случае 'partner') - уникальный ID сущности данных
    2) num_cols (если есть) - Список списков (каждый список - значение одного конкретного признака во всех клиентских привлечениях)
    3) cat_cols (если есть) - Список списков (каждый список - значение одного конкретного признака во всех клиентских привлечениях)
    4) fixed_num_cols (если есть) - Список
    5) fixed_cat_cols (если есть) - Список
    

    :param actions_frame: pd.DataFrame (фрейм с привлечениями партнера
    :param num_last_actions: int (количество последних привлечений клиентов, которые будут рассмотрены)
    :param di_features: dict (словарь с фичами)
    
    :return: pd.DataFrame, число строк = число партнеров, число столбцов - от 2 до 5
    """
    
    instance_col = di_features['instance_col'] 
    time_col = di_features['time_col'] 
    
    if di_features['has_num']:
        out = actions_frame \
            .sort_values([instance_col, time_col, 'client']) \
            .groupby([instance_col])[di_features['num_cols']] \
            .apply(lambda x: truncate(x, num_last_actions=num_last_actions)) \
            .reset_index().rename(columns={0: 'num_cols'})
        
        if di_features['has_cat']:
            out['cat_cols'] = actions_frame \
            .sort_values([instance_col, time_col, 'client']) \
            .groupby([instance_col])[di_features['cat_cols']] \
            .apply(lambda x: truncate(x, num_last_actions=num_last_actions)).values
            
    else:
        out = actions_frame \
            .sort_values([instance_col, time_col, 'client']) \
            .groupby([instance_col])[di_features['cat_cols']] \
            .apply(lambda x: truncate(x, num_last_actions=num_last_actions)) \
  
        
    if di_features['has_fixed_num']:
        out['fixed_num_cols'] = actions_frame.drop_duplicates(subset=instance_col)[di_features['fixed_num_cols']]\
         .apply(lambda x: x.values.transpose().tolist(), axis=1).values
    
    if di_features['has_fixed_cat']:
        out['fixed_cat_cols'] = actions_frame.drop_duplicates(subset=instance_col)[[instance_col] + di_features['fixed_cat_cols']]\
         .set_index(instance_col).loc[out['partner'].values].apply(lambda x: x.values.transpose().tolist(), axis=1).values
    
    
    return out


def create_padded_buckets(frame_of_sequences: pd.DataFrame, bucket_info: Dict[int, int],
                          save_to_file_path=None, has_target=True, di_features=None):
    
    
    """
    Функция реализует sequence_bucketing технику для обучения нейронных сетей.
    Принимает на вход frame_of_sequences (результат работы функции transform_actions_to_sequences),
    словарь bucket_info, где для последовательности каждой длины указано, до какой максимальной длины нужно делать
    padding, далее группирует транзакции по бакетам (на основе длины), делает padding транзакций и сохраняет результат
    в pickle файл, если нужно
    
    :param frame_of_sequences: pd.DataFrame (c транзакциями, результат применения transform_actions_to_sequences)
    :param bucket_info: dict (словарь, где для последовательности каждой длины указано, до какой максимальной длины нужно делать
    padding)
    :param save_to_file_path: str, optional (путь к файлу, куда нужно сохранить результат)
    :param has_target: bool (Есть ли таргет)
    :param di_features: dict (Словарь с переменными)
    :return: dict (Бакет)
    """
    
    instance_col = di_features['instance_col']
    target_col = di_features['target_col']
    
    frame_of_sequences['bucket_idx'] = frame_of_sequences.sequence_length.map(bucket_info)
    
    num_values = []
    cat_values = []
    targets = []
    ids = []
    fixed_num_values = []
    fixed_cat_values = []
    partitions = []


    for size, bucket in frame_of_sequences.groupby('bucket_idx'):
    # for size, bucket in tqdm(frame_of_sequences.groupby('bucket_idx'), desc='Extracting buckets'):
        
        ids.append(bucket[instance_col].values)
        
        if di_features['has_num']:

            padded_num_cols = bucket['num_cols'].apply(lambda x: pad_sequence(x, size)).values
            padded_num_cols = [np.array(x) for x in padded_num_cols]
            num_values.append(padded_num_cols)
        
        
        if di_features['has_cat']:
            padded_cat_cols = bucket['cat_cols'].apply(lambda x: pad_sequence(x, size)).values
            padded_cat_cols = [np.array(x) for x in padded_cat_cols]
            cat_values.append(padded_cat_cols)

        if has_target:
            targets.append(bucket[target_col].values)
        
        if di_features['has_fixed_num']:
            fixed_num_values.append(bucket['fixed_num_cols'].values.tolist())
            
        if di_features['has_fixed_cat']:
            fixed_cat_values.append(bucket['fixed_cat_cols'].values.tolist())
            
        if 'partition' in bucket:
            partitions.append(bucket['partition'].values)
        

    frame_of_sequences.drop(columns=['bucket_idx'], inplace=True)


    
    dict_result = {}
    dict_result['ids'] = ids
    if num_values:
        dict_result['num_values'] = num_values
    if cat_values:
        dict_result['cat_values'] = cat_values
    if fixed_num_values:
        dict_result['fixed_num_values'] = fixed_num_values
    if fixed_cat_values:
        dict_result['fixed_cat_values'] = fixed_cat_values
    if partitions:
        dict_result['partitions'] = partitions
    if targets:
        dict_result['targets'] = targets
    
    
    if save_to_file_path:
        with open(save_to_file_path, 'wb') as f:
            pickle.dump(dict_result, f)
    return dict_result



def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0,
                                     num_parts_to_read: int = 2, verbose=False) -> pd.DataFrame:
    """
    читает num_parts_to_read партиций, преобразует их к pd.DataFrame и возвращает
    :param path_to_dataset: str (путь до директории с партициями)
    :param start_from: int (номер партиции, с которой начать чтение)
    :param num_parts_to_read: int (количество партиций, которые требуется прочитать)
    :return: pd.DataFrame
    """

    res = []
    dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset) 
                              if filename.startswith('part')])
    
    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]
    if verbose:
        print('Reading chunks:\n')
        for chunk in chunks:
            print(chunk)
    # for chunk_path in tqdm(chunks, desc="Reading dataset with pandas"):
    for chunk_path in chunks:
        chunk = pd.read_parquet(chunk_path)
        res.append(chunk)
    return pd.concat(res).reset_index(drop=True)
