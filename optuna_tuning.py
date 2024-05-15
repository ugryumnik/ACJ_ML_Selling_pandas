# Находим лучшие параметры с помощью optuna

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import optuna
import torch

from sklearn.metrics import roc_auc_score


def tune_hyperparams(
    X_train, X_val, y_train, y_val, model, #cat_cols
    n_trials, save_weights=True, fixed_params=None
):
    
    if fixed_params is None:
        fixed_params = {}
    

    if model not in ["catboost", "xgboost", "lightgbm"]:
        print(f"Incorrect model {model}, valid are ['catboost', 'xgboost', 'lightgbm']")
        


    if model == "catboost":
            
        base_params =  {"n_estimators": 1000,
                            "verbose": 50,
                            "early_stopping_rounds": 50,
                            "loss_function": "Logloss",
                            "eval_metric": "AUC",
                            "random_state": 42, 
                            "use_best_model": True}       # Базовые параметры, необходимые для модели
        if torch.cuda.is_available():
            task_type='GPU'
        else:
            print('GPU IS NOT AVAILABLE!')
            task_type='CPU'
        base_params['task_type']=task_type
        
        
        for i in base_params.keys():
            if i not in fixed_params:
                fixed_params[i]=base_params[i]
            
        def objective(
            trial, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, init_params=fixed_params
        ):
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.02, 0.2, log=True
                ),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", 1e-2, 100.0, log=True
                ),
                "random_strength": trial.suggest_float(
                    "random_strength", 1e-4, 10.0, log=True
                )
            }

            for i in init_params.keys():
                params[i]=init_params[i]
                

            model_c = CatBoostClassifier(
                **params
            )
            


            model_c.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            preds = model_c.predict_proba(X_val.values)[:, 1]
            auc = roc_auc_score(y_val, preds)
            
            # if save_weights:
            #     file_name = f"checkpoints_boostings/catboost_{gini_x:.4f}.bin"
            #     model_c.save_model(file_name)
                # pickle.dump(model_x, open(file_name, "wb"))

            return auc

    elif model == "xgboost":
        base_params = {  "n_estimators": 1000,
                            "eval_metric": "auc",
                            "early_stopping_rounds": 50,
                            "random_state": 42,
                            # "enable_categorical": True,
                      }         # Базовые параметры, необходимые для модели
        
        if torch.cuda.is_available():
                task_type='gpu_hist'
        else:
            print('GPU IS NOT AVAILABLE!')
            task_type='hist'
        base_params['tree_method']=task_type
        
        for i in base_params.keys():
            if i not in fixed_params:
                fixed_params[i]=base_params[i]
                
            
        def objective(
            trial, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, init_params=fixed_params
        ):
            params = {
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            }
            
            for i in init_params.keys():
                params[i]=init_params[i]

                    
            model_x = XGBClassifier(**params)
    
            model_x.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
            preds = model_x.predict_proba(X_val.values)[:, 1]
            auc = roc_auc_score(y_val, preds)
        
            
            # if save_weights:
            #     file_name = f"checkpoints_boostings/xgboost_{gini_x:.4f}.pkl"
            #     pickle.dump(model_x, open(file_name, "wb"))

            return auc

          

    elif model == "lightgbm":
        base_params = {
                                "objective": "binary",
                                "n_estimators": 1000,
                                "early_stopping_round": 50,
                                "random_state": 42,
                                "verbosity": -1,
                            }   # Базовые параметры, необходимые для модели
        
        for i in base_params.keys():
            if i not in fixed_params:
                fixed_params[i]=base_params[i]
        
        def objective(
            trial, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, init_params=fixed_params
        ):
            params = {
                    "subsample": trial.suggest_float("subsample", 0.5, 1),
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0),
                    "num_leaves": trial.suggest_int("num_leaves", 32, 200),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                    "min_child_samples": trial.suggest_int("min_child_samples", 25, 200),
                }
            

            for i in init_params.keys():
                params[i]=init_params[i]


            model_l = LGBMClassifier(**params)
            
            model_l.fit(X_train, y_train, eval_set=[(X_val, y_val.values.squeeze())], verbose=-1, eval_metric='auc')
            preds = model_l.predict_proba(X_val.values)[:, 1]
            auc = roc_auc_score(y_val, preds)

            return auc

            
    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    print('='*30)
    trial = study.best_trial

    print("  AUC: {}".format(trial.value))
    print('='*30)
    print("  Params: {")
    
    params = trial.params
    
    for i in fixed_params.keys():
            if i not in params:
                params[i]=fixed_params[i]

    for key, value in params.items():
        if isinstance(value, str):
            print("    '{}': '{}',".format(key, value))
        else:
            print("    '{}': {},".format(key, value))
    print('}')
    return params, trial.value