# Решение задачи "Предсказание оттока партнеров", хакатон Alfa Campus Junior 2024

Требовалось по данным о привлечениях клиентов партнерами с 2019-03-01 по 2020-11-31 предсказать, привлечет ли каждый из партнеров еще одного клиента в период с 2020-12-01 по 2021-02-28.  

В решении использовались две модели: LightGBM и LSTM.

## Структура репозитория:

Многие .py файлы дублируют название .ipynb файлов. Это просто ноутбуки, переведенные в .py файлы для удобства воспроизведения кода. Для изучения кода же удобнее использовать соответствующие ноутбуки.


base_processing.ipynb - Базовая предобработка датасета, необходимая для обеих моделей  
blending.ipynb - Ноутбук с блендингом моделей  
boosting_processing.ipynb - Ноутбук с предобработкой датасета для бустинга  
boosting_training.ipynb - Ноутбук с обучением бустинга  
data_generators.py - Функция для генерации батчей на вход LSTM  
inference.ipynb - Ноутбук с инференсом моделей  
optuna_tuning.py - Функция для тюнинга параметров бустинга  
pytorch_training.py - Функции для обучения, валидации и инференса LSTM  
requirements.txt - Требуемые библиотеки  
rnn_model.py - Архитектура модели на основе LSTM  
rnn_processing.py - Ноутбук с предобработкой датасета для LSTM  
rnn_training.py - Ноутбук с обучением LSTM  
rnn_utils.py - Некоторые полезные функции для LSTM (в основном, для обработки данных)  

## Результаты
LSTM: 77.18% AUC  
LGBMClassifier: 77.51% AUC  
BLENDED (0.375 * LSTM + 0.625 * LGBM): 77.65% AUC  

## Запуск моделей
Клонируйте репозиторий:
```
git clone https://github.com/ugryumnik/ACJ_ML_Selling_pandas.git 
cd ACJ_ML_Selling_pandas
```
Установите необходимые библиотеки:
```
pip install -r requirements.txt
```

Далее есть два способа:  
### 1. Инференс (подгружаются готовые веса)
Необходимо запустить следующую команду:  
```
python inference.py
```
Предсказания будут сохранены в папку submissions (сейчас она пустая).  
"rnn.csv" - предсказания LSTM  
"lightgbm.csv" - предсказания бустинга  
"blended_lgbm_rnn.csv" - блендинг двух моделей  

### 2. Полный цикл обучения
Чтобы убедиться в достоверности, можете удалить все папки, кроме "data" (там должны остаться файлы "dataset.csv" и "submission_example.csv")
```
ipython base_processing.py
ipython boosting_processing.py
ipython rnn_processing.py
ipython boosting_training.py
ipython rnn_training.py
ipython blending.py
```
Через 2-3 минуты предсказания будут сохранены в папку submissions.  
"rnn.csv" - предсказания LSTM  
"lightgbm.csv" - предсказания бустинга  
"blended_lgbm_rnn.csv" - блендинг двух моделей  




Общая логика и значительная часть кода для обучения LSTM заимствованы из данного репозитория: https://github.com/smirnovevgeny/AlfaBattle2.0



