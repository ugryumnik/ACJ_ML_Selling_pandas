import numpy as np
import pickle
import torch



def batches_generator(list_of_paths, batch_size=32, shuffle=False,
                      verbose=False, has_target=True, di_features=None):
    
    
    """
    функция для создания батчей на вход для нейронной сети
    :param list_of_paths: list of strings (пути к бакетам)
    :param batch_size: int (размер батча)
    :param shuffle: bool (перемешивать ли данные: бакеты и последовательности внутри одного бакета)
    :param verbose: bool (если True, то печатает текущий обрабатываемый файл)
    :param has_target: bool (Есть ли таргет)
    :param di_features: dict (Словарь с переменными)
    :return: dict (батч с данными)
    """

    if shuffle:
        list_of_paths = np.array(list_of_paths)
        if len(list_of_paths) > 1:
            list_of_paths = list_of_paths[torch.randperm(len(list_of_paths))] # Перемешиваем пути к бакетам

    for path in list_of_paths:
        if path.endswith('checkpoints'): continue
        if verbose:
            print(f'reading {path}')

        with open(path, 'rb') as f:
            data = pickle.load(f)

        ids = data['ids']
        num_values, cat_values, fixed_num_values, fixed_cat_values, targets, partitions \
        = None, None, None, None, None, None


        if di_features['has_num']:
            num_values = data['num_values']
        if di_features['has_cat']:
            cat_values = data['cat_values']
        if di_features['has_fixed_num']:
            fixed_num_values = data['fixed_num_values']
        if di_features['has_fixed_cat']:
            fixed_cat_values = data['fixed_cat_values']
        if has_target:
            targets = data['targets']


        if 'partitions' in data.keys():
            partitions = data['partitions']


        indices = np.arange(len(ids))

        if shuffle:
            indices = indices[torch.randperm(len(indices))]
            ids = [ids[i] for i in indices]

            if num_values:
                num_values = [num_values[i] for i in indices]
            if cat_values:
                cat_values = [cat_values[i] for i in indices]

            if fixed_num_values:
                fixed_num_values = [fixed_num_values[i] for i in indices]

            if fixed_cat_values:
                fixed_cat_values = [fixed_cat_values[i] for i in indices]

            if partitions:
                partitions = [partitions[i] for i in indices]
            if has_target:
                targets = [targets[i] for i in indices]



        for idx in range(len(ids)):
            ids_one_length = ids[idx]

            if num_values:
                num_values_one_length = num_values[idx]

            if cat_values:
                cat_values_one_length = cat_values[idx]

            if fixed_num_values:
                fixed_num_values_one_length = fixed_num_values[idx]

            if fixed_cat_values:
                fixed_cat_values_one_length = fixed_cat_values[idx]


            if partitions:
                partitions_one_length = partitions[idx]

            if has_target:
                targets_one_length = targets[idx]



            for jdx in range(0, len(num_values_one_length), batch_size):

                batch = dict()

                batch_ids = ids_one_length[jdx: jdx + batch_size]
                batch['ids'] = batch_ids # Id партнеров

                if num_values:
                    batch_num_values = num_values_one_length[jdx: jdx + batch_size]
                    batch['num_values'] = batch_num_values # Числовые фичи

                if cat_values:
                    batch_cat_values = cat_values_one_length[jdx: jdx + batch_size]
                    batch['cat_values'] = batch_cat_values # Категориальные фичи

                if fixed_num_values:
                    batch_fixed_num_values = fixed_num_values_one_length[jdx: jdx + batch_size]
                    batch['fixed_num_values'] = batch_fixed_num_values # Фиксированные числовые фичи

                if fixed_cat_values:
                    batch_fixed_cat_values = fixed_cat_values_one_length[jdx: jdx + batch_size]
                    batch['fixed_cat_values'] = batch_fixed_cat_values # Фиксированные категориальные фичи

                if partitions:
                    batch_partitions = partitions_one_length[jdx: jdx + batch_size]
                    batch['partitions'] = batch_partitions # Партиции

                if has_target:
                    batch_targets = targets_one_length[jdx: jdx + batch_size]
                    batch['targets'] = batch_targets # Таргеты



                yield batch
