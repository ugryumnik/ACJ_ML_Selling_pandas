import torch
import torch.nn as nn

class ChurnPredictor(nn.Module):
    def __init__(self, di_features, embedding_projections, rnn_units=128, 
                spatial_dropout=5e-2,
                 use_embeds_pooling=True, is_bidirectional=True):
        super(ChurnPredictor, self).__init__()
        
        self.use_embeds_pooling = use_embeds_pooling # Брать ли пулинги от эмбеддингов
        
        self.di_features = di_features # Cловарь с переменными
        
        self.num_features = di_features['num_cols']
        self.cat_features = di_features['cat_cols']
        self.fixed_num_features = di_features['fixed_num_cols']
        self.fixed_cat_features = di_features['fixed_cat_cols']
        
        self.num_embeddings_len = sum([embedding_projections[x][1] for x in self.num_features])
        self.cat_embeddings_len = sum([embedding_projections[x][1] for x in self.cat_features])
        self.fixed_num_embeddings_len = sum([embedding_projections[x][1] for x in self.fixed_num_features])
        self.fixed_cat_embeddings_len = sum([embedding_projections[x][1] for x in self.fixed_cat_features])
        
        self.num_embeddings = nn.ModuleList([self.create_embedding_projection(*embedding_projections[feature]) 
                                                for feature in self.num_features])
        self.cat_embeddings = nn.ModuleList([self.create_embedding_projection(*embedding_projections[feature]) 
                                                for feature in self.cat_features])
        
        self.non_fixed_embeddings_len = self.num_embeddings_len + self.cat_embeddings_len
        
        self.fixed_num_embeddings = nn.ModuleList([self.create_embedding_projection(*embedding_projections[feature]) 
                                                for feature in self.fixed_num_features])
        self.fixed_cat_embeddings = nn.ModuleList([self.create_embedding_projection(*embedding_projections[feature]) 
                                                for feature in self.fixed_cat_features])
        self.fixed_embeddings_len = self.fixed_num_embeddings_len + self.fixed_cat_embeddings_len
        
        
        
        
        self.spatial_dropout = nn.Dropout2d(spatial_dropout) # Spatial dropout для эмбеддингов
        
        self.rnn = nn.LSTM(input_size=self.num_embeddings_len + self.cat_embeddings_len,
                             hidden_size=rnn_units, batch_first=True, bidirectional=is_bidirectional) # Рекуррентный слой
        
        
        self.linear = nn.Linear(in_features=rnn_units * (2 + 2 * is_bidirectional) + (2 * use_embeds_pooling * self.non_fixed_embeddings_len) + self.fixed_embeddings_len, 
                                         out_features=1) # Линейный слой поверх над выходом LSTM
        
        self.sigmoid=nn.Sigmoid()
        
    
    def forward(self, num_values, cat_values, fixed_num_values, fixed_cat_values):
        

        fixed_embeddings = None
        
        
        # Эмбеддинги нефиксированных фичей
        if self.di_features['has_num'] and self.di_features['has_cat']:
            num_embeddings = [embedding(num_values[:, i] + 1) for i, embedding in enumerate(self.num_embeddings)]
            cat_embeddings = [embedding(cat_values[:, i] + 1) for i, embedding in enumerate(self.cat_embeddings)]
            concated_embeddings = torch.cat(num_embeddings + cat_embeddings, dim=-1)
        elif self.di_features['has_num']:
            num_embeddings = [embedding(num_values[:, i] + 1) for i, embedding in enumerate(self.num_embeddings)]
            concated_embeddings = torch.cat(num_embeddings, dim=-1)
        else:
            cat_embeddings = [embedding(cat_values[:, i] + 1) for i, embedding in enumerate(self.cat_embeddings)]
            concated_embeddings = torch.cat(cat_embeddings, dim=-1)
            

        concated_embeddings = self.spatial_dropout(concated_embeddings) # Применяем spatial dropout для нефиксированных эмбеддингов

        rnn_output = self.rnn(concated_embeddings)[0] # Получаем скрытые состояния из LSTM

        avg_pool = torch.mean(rnn_output, 1) # Mean pooling над скрытыми состояниями
        max_pool, _ = torch.max(rnn_output, 1) # Max pooling над скрытыми состояниями

        
        # Эмбеддинги фиксированных фичей
        if self.di_features['has_fixed_num'] and self.di_features['has_fixed_cat']:
            fixed_num_embeddings = [embedding(fixed_num_values[:, i]) for i, embedding in enumerate(self.fixed_num_embeddings)]
            fixed_cat_embeddings = [embedding(fixed_cat_values[:, i]) for i, embedding in enumerate(self.fixed_cat_embeddings)]
            fixed_embeddings = fixed_num_embeddings + fixed_cat_embeddings
        elif self.di_features['has_fixed_num']:
            fixed_embeddings = [embedding(fixed_num_values[:, i]) for i, embedding in enumerate(self.fixed_num_embeddings)]
        elif self.di_features['has_fixed_cat']:
            fixed_embeddings = [embedding(fixed_cat_values[:, i]) for i, embedding in enumerate(self.fixed_cat_embeddings)]
            
        if fixed_embeddings:
            fixed_embeddings = torch.cat(fixed_embeddings, dim=-1)
            
            
        # Объединяем пулинги над скрытыми состояними, эмбеддинги фиксированных фичей и пулинг над эмбеддингами (опционально)
        if self.use_embeds_pooling: # Если применяем пулинги к нефиксированным эмбеддингам
            avg_pool_embeds = torch.mean(concated_embeddings, 1)
            max_pool_embeds, _ = torch.max(concated_embeddings, 1)
            last_hidden = torch.cat((avg_pool, max_pool, avg_pool_embeds, max_pool_embeds, fixed_embeddings), 1)
        else:
            last_hidden = torch.cat((avg_pool, max_pool, fixed_embeddings), 1)
  
        prob = self.sigmoid(self.linear(last_hidden))
    
        return prob
        
    @classmethod
    def create_embedding_projection(cls, cardinality, embed_size, add_missing=True):
        
        """
        Создаем эмбеддинг нужной размерности
        :param cardinality: int (Число уникальных значений фичи)
        :param embed_size: Размер эмбеддинга
        :param add_missing: Сохранить ли еще один слот для неизвестных значений
        """
        
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=None)