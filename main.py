import time
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from model import cv_model, train_model, get_class_weights, get_important_features, save_predictions
from preprocessing import get_shuffle_splits, get_bootstrap
from catboost import CatBoostClassifier

"""
Argument Parsing
"""
parser = argparse.ArgumentParser(description='ICLR Model Training Script')

parser.add_argument('-cv', '--cross_validation', help='Perform 5 fold cross-validation', action='store_true')
parser.add_argument('-sp', '--submission_path', default=time.asctime(), type=str, 
    help='File path for saving Zindi submission file. Defaults to current time.')

model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('-md', '--model', default='RandomForest', type=str,
    help=f'model for training and prediction. Accepts CatBoost, RandomForest, ExtraTrees.')
model_params.add_argument('-ne', '--n_estimators', help='number of estimators for ensemble model.', default=500, type=int)
model_params.add_argument('-rs', '--random_seed', help='Random seed value for reproducibility.', default=123, type=int)
model_params.add_argument('-fe', '--important_features', type=str,
    help='List of features to pass for model training stored as line separated text file.')

parser.add_argument('-bs', '--bootstrap', help='bootstrap training data', action='store_true')

agg_data = parser.add_mutually_exclusive_group()
agg_data.add_argument('-px', '--pixels', action='store_true', help='Train on pixel-level data.')
agg_data.add_argument('-fd', '--field', action='store_true', help='Train on mean-aggregated field data.')

args = parser.parse_args()

"""
Loading Data, preparing script parameters
"""
print(f'Model training with {args}')
print(f'Currrent time: {time.asctime()}')

data = pd.read_csv('./csv_data/train_stat_features.csv')
zindi = pd.read_csv('./csv_data/test_stat_features.csv')

"""
Use pre-determined important features
"""
if args.important_features:
    with open(args.important_features, 'r') as f:
        impo_features = f.read().split('\n')[:-1]

    impo_features.append('field_id')
    impo_features.append('label')

    data = data.loc[:, impo_features]
    zindi = zindi.loc[:, impo_features]

if args.pixels:
    print('Training on individual pixel values.')
else:
    print('Training on field aggregated values.')
    data = data.groupby('field_id', as_index=False).mean()
    zindi = zindi.groupby('field_id', as_index=False).mean()

class_weights = get_class_weights(data)

"""
Instantiate model
"""
if args.model == 'RandomForest':
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        n_jobs=-1,
        class_weight=class_weights,
        random_state=args.random_seed
    )
elif args.model == 'CatBoost':
    model = CatBoostClassifier(
        iterations=args.n_estimators,
        random_seed=args.random_seed,
        class_weights=class_weights,
        verbose=0
    )
elif args.model == 'ExtraTrees':
    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        n_jobs=-1,
        class_weight=class_weights,
        random_state=args.random_seed
    )

"""
Cross validation
"""
if args.cross_validation: 
    print('Begin 5-fold cross validation...\n', '-' * 79)
    splits = get_shuffle_splits(
        data['field_id'].unique(),
        random_seed=args.random_seed
    )

    train_scores = []
    test_scores = []
    
    for split in tqdm(splits, total=5):
        train_fields, test_fields = split
        train_filter = data['field_id'].isin(train_fields)
        test_filter = data['field_id'].isin(test_fields)

        train = data.loc[train_filter, :]
        test = data.loc[test_filter, :]

        if args.bootstrap:
            train = get_bootstrap(
                    train,
                    random_seed=args.random_seed
                )
    
        train_score, test_score = cv_model(
            model,
            train,
            test
        )

        train_scores.append(train_score)
        test_scores.append(test_score)

    print('CV Results - Log Loss')
    print('_' * 72)
    print('     Train | Test')
    for i in range(len(train_scores)):
        print(f'{i + 1}: {round(train_scores[i], 4)}  | {round(test_scores[i], 4)}')
    print(f'\nK-folds CV scores mean: {round(np.mean(test_scores), 4)}')
    print(f'K-folds CV scores std: {round(np.std(test_scores), 4)}')

"""
Re-train model and save predictions to submission path
"""
if args.bootstrap:
    data = get_bootstrap(
        data,
        random_seed=args.random_seed
    )

model = train_model(
    data,
    model   
)

save_predictions(
    model,
    zindi,
    args.submission_path
)
