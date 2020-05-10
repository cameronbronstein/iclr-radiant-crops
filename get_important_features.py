import argparse
import pandas as pd 
from model import get_important_features, get_class_weights
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

parser = argparse.ArgumentParser('Get Important Features')
parser.add_argument('-ni', '--n_iterations', default=100, type=int,
    help='Number of model iterations to run to aggregate feature importance values.')
parser.add_argument('-nf', '--n_features', default=100, type=int, choices=[range(1, 360)],
    help='Percentage of original feature to return, ordered by feature importance value.')

parser.add_argument_group('Model Parameters')      
parser.add_argument('-md', '--model', help='RandomForest or CatBoostClassifier', type=str)
parser.add_argument('-ne', '--n_estimators', type=int, help='Number of estimators in each ensemble model.')
parser.add_argument('-rs', '--random_seed', help='Random seed value for reproducibility.', default=123, type=int)

agg_data = parser.add_mutually_exclusive_group()
agg_data.add_argument('-px', '--pixels', action='store_true', help='Train on pixel-level data.')
agg_data.add_argument('-fd', '--field', action='store_true', help='Train on mean-aggregated field data.')

args = parser.parse_args()

data = pd.read_csv('./csv_data/train_stat_features.csv')

class_weights = get_class_weights(data)

if args.model == 'RandomForest':
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight=class_weights,
        n_jobs=-1
    )
elif args.model == 'CatBoost':
    model = CatBoostClassifier(
        iterations=args.n_estimators,
        class_weights=class_weights,
        verbose=0
    )

if args.pixels:
    data = data.groupby('field_id', as_index=False).mean()

impo_features = get_important_features(
    X = data.drop(['field_id', 'label'], axis=1),
    y = data['label'],
    estimator=model,
    n_iterations=args.n_iterations,
    random_seed=args.random_seed,
    n_features=args.n_features
)

for feature in impo_features:
    print(feature)