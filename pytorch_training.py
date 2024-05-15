from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score
import pandas as pd

criterion = torch.nn.BCELoss()


def train_epoch(model, dataloader_train, cfg, optimizer, scheduler, epoch):
    """
    Выполняет одну эпоху обучения модели
    :param model: nn.Module (модель)
    :param dataloader_train: DataLoader (выход из batches_generator)
    :param cfg: Конфиг обучения
    :param optimizer: nn.optim (оптимизатор)
    :param scheduler: nn.optim.lr_scheduler (шкедулер)
    :param epoch: int (Номер эпохи)
    :return: int (ROC AUC SCORE)
    """
    device = cfg.device
    di_features = cfg.di_features
    
    
    predicted_labels = []
    true_labels = []
    
    for count_batch, batch in enumerate(tqdm(dataloader_train)):

        batch = {k: torch.tensor(v).to(device) for k, v in batch.items()}
        
        ids=batch['ids']
        num_values, cat_values, fixed_num_values, fixed_cat_values = None, None, None, None
        targets = batch['targets']
        
        if di_features['has_num']:
            num_values = batch['num_values']
        if di_features['has_cat']:
            cat_values = batch['cat_values']
        if di_features['has_fixed_num']:
            fixed_num_values = batch['fixed_num_values']
        if di_features['has_fixed_cat']:
            fixed_cat_values = batch['fixed_cat_values']
            
        
        
        optimizer.zero_grad()
        
        preds = model(num_values, cat_values, fixed_num_values, fixed_cat_values).reshape(-1)
        
        loss = criterion(preds, targets.float())

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if batch['partitions'][0].item() == 0: # Только для не аугментированной части трейна считаем метрику
            true_labels.extend(targets.cpu().numpy())
            predicted_labels.extend(preds.detach().cpu().numpy())
        
    auc = roc_auc_score(true_labels, predicted_labels)
    print('-'*50)
    print('Epoch: {}'.format(epoch))
    print('Train AUC: {}'.format(auc))
    
    
def validate(model, dataloader_val, cfg):
    
    """
    Выполняет валидацию
    :param model: nn.Module (модель)
    :param dataloader_val: DataLoader (выход из batches_generator)
    :param cfg: Конфиг обучения
    """
    
    device = cfg.device
    di_features = cfg.di_features
    
    predicted_labels = []
    true_labels = []
        
    for count_batch, batch in enumerate(dataloader_val):

        batch = {k: torch.tensor(v).to(device) for k, v in batch.items()}

        ids=batch['ids']
        num_values, cat_values, fixed_num_values, fixed_cat_values = None, None, None, None
        targets = batch['targets']
        
        if di_features['has_num']:
            num_values = batch['num_values']
        if di_features['has_cat']:
            cat_values = batch['cat_values']
        if di_features['has_fixed_num']:
            fixed_num_values = batch['fixed_num_values']
        if di_features['has_fixed_cat']:
            fixed_cat_values = batch['fixed_cat_values']


        with torch.no_grad():


            preds = model(num_values, cat_values, fixed_num_values, fixed_cat_values).reshape(-1)


            true_labels.extend(targets.cpu().numpy())
            predicted_labels.extend(preds.detach().cpu().numpy())


    auc = roc_auc_score(true_labels, predicted_labels)
    print('Validation AUC: {}'.format(auc))
    return auc

def inference(model, dataloader_test, device, di_features, path_to_sample_submission, path_to_save):
    
    """
    Выполняет инференс
    :param model: nn.Module (модель)
    :param dataloader_test: DataLoader (выход из batches_generator)
    :param device: torch.device (На каком устройстве проводить обучение)
    :param di_features: Словарь с переменными
    :param path_to_sample_submission: Путь к sample submission
    :param path_to_save: Куда сохранить предсказания
    """
    
    
    model.eval()
    
    all_ids = []
    predicted_labels = []


    for batch in dataloader_test:
        
        all_ids.extend(batch['ids'])
        batch = {k: torch.tensor(v).to(device) for k, v in batch.items()}
        ids=batch['ids']
        num_values, cat_values, fixed_num_values, fixed_cat_values = None, None, None, None
        
        if di_features['has_num']:
            num_values = batch['num_values']
        if di_features['has_cat']:
            cat_values = batch['cat_values']
        if di_features['has_fixed_num']:
            fixed_num_values = batch['fixed_num_values']
        if di_features['has_fixed_cat']:
            fixed_cat_values = batch['fixed_cat_values']


        with torch.no_grad():
            preds = model(num_values, cat_values, fixed_num_values, fixed_cat_values).reshape(-1)
            predicted_labels.extend(preds.detach().cpu().numpy())


    sample_submission = pd.read_csv(path_to_sample_submission)
    
    submit = sample_submission.copy()
    submit = submit.set_index('clientbankpartner_pin')
    submit.loc[all_ids, 'score'] = predicted_labels
    submit['score'] = 1 - submit['score'] # Изначально при обучении я перепутал классы, поэтому здесь инвертирую вероятности
    submit = submit.reset_index()
    
    submit.to_csv(path_to_save, index=False)
    return submit