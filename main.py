import time
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from model import cv_model, train_model, get_class_weights, get_important_features
from predict import save_predictions
from preprocessing import get_shuffle_splits, get_bootstrap

parser = argparse.ArgumentParser(description='ICLR Model Training Script')

parser.add_argument('-cv', '--cross_validation', help='Perform 5 fold cross-validation', default=False, type=bool)
parser.add_argument('-bs', '--bootstrap', help='bootstrap training data', default=True, type=bool)
parser.add_argument('-px', '--pixels', 
    help='Train on pixel-level or aggregate field-level data. Accepts: False --> fields; True --> pixels',
    default=False, type=bool)

parser.add_argument('-md', '--model', 
    help=f'model for training and prediction. Accepts CatBoost, RandomForest, ExtraTrees.', 
    default='RandomForest', type=str)

parser.add_argument('-ne', '--n_estimators', help='number of estimators for ensemble model.', default=500, type=int)

parser.add_argument('-rs', '--random_seed', help='Random seed value for reproducibility.', default=123, type=int)
parser.add_argument('-sp', '--save_path', help='File path for saving Zindi submission file.', default=time.asctime(), type=str)

args = parser.parse_args()

print(f'Model training with {args}')
print(f'Currrent time: {time.asctime()}')

# load data
data = pd.read_csv('./csv_data/train_stat_features.csv')
zindi = pd.read_csv('./csv_data/test_stat_features.csv')

if args.pixels == True:
    print('Training on individual pixel values.')
else:
    print('Training on field aggregated values.')
    data = data.groupby('field_id', as_index=False).mean()
    zindi = zindi.groupby('field_id', as_index=False).mean()

class_weights = get_class_weights(data)

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
        random_seed=123,
        verbose=100
    )
elif args.model == 'ExtraTrees':
    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        n_jobs=-1,
        class_weight=class_weights,
        random_state=args.random_seed
    )

impo_features = get_important_features(
    X = data.drop(['field_id', 'label'], axis=1),
    y = data['label'],
    estimator=RandomForestClassifier(
        n_estimators=1,
        class_weight=class_weights
    ),
    n_iterations=10,
    random_seed=123,
    n_features=100
)

impo_features.append('field_id')
impo_features.append('label')

data = data.loc[:, impo_features]
zindi = data.loc[:, impo_features]

if args.cross_validation == True: 
    print('Begin 5-fold cross validation...\n', '-' * 79)
    splits = get_shuffle_splits(
                data['field_id'].unique()
                )

    cv_scores = []
    
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
    
        split_score = cv_model(
            model,
            train,
            test
        )

        cv_scores.append(split_score)

    print('Model evaluation: log-loss')
    for iter, score in enumerate(cv_scores):
        print(f'CV {iter + 1} validation loss: {score}')
    print(f'K-folds CV scores mean: {np.mean(cv_scores)}')
    print(f'K-folds CV scores std: {np.std(cv_scores)}')

if args.bootstrap == True:
    data = get_bootstrap(data)

model = train_model(
    data,
    model   
)

save_predictions(
    model,
    zindi,
    args.save_path
)
