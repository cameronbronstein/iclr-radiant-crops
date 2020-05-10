import pandas as pd
import numpy as np
from tqdm import tqdm
import time

from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_class_weight

def cv_model(
    estimator,
    train,
    test,
    random_seed=123,
    target_col='label'
):
    """
    Function to fit model with cross validation
    
    Params
    -------
    data: 
    estimator:
    target_col: 

    Returns
    -------
    Field aggregated validation loss

    Returns training and validation log-loss to standard output.
    """
    y_train = train.loc[:, target_col]  
    X_train = train.drop([target_col, 'field_id'], axis=1)
    
    y_test = test.loc[:, target_col]
    X_test = test.drop([target_col, 'field_id'], axis=1)

    estimator.fit(X_train, y_train)

    train_pred = estimator.predict_proba(X_train)
    test_pred = estimator.predict_proba(X_test)
    
    train_score = log_loss(y_train, train_pred, labels=y_train.unique())
    test_score = log_loss(y_test, test_pred, labels=y_train.unique()) 
          
    return train_score, test_score

def train_model(
    data,
    estimator,
    target_col='label'
):
    X = data.drop([target_col, 'field_id'], axis=1)
    y = data.loc[:, target_col]

    estimator.fit(X, y)
    
    return estimator

def get_class_weights(
    data,
    target_col='label'
):
    """
    Get class weight for weighted RandomForest algorithm
    """
    weights = compute_class_weight(
        class_weight='balanced',
        classes=data[target_col].unique(),
        y=data[target_col],
    )

    class_weights = dict(
            zip(data[target_col].unique(),
                weights)
            )

    return class_weights

def save_predictions(
    fit_estimator, 
    zindi, 
    output_filename=time.asctime()
):
    """
    Function to calculate predictions on the hold-out data

    Params
    ------
        - fit_estimator
        - zindi: the test dataset.
        - output_fn: string filename to save predictions
    """
    preds = fit_estimator.predict_proba(zindi.drop(['field_id', 'label'], axis=1))
    
    preds_df = pd.DataFrame(
        preds,
        index=zindi['field_id'],
        columns=range(1,8)
    )     

    preds_df = preds_df.groupby('field_id').mean()

    preds_df.to_csv(f'./submissions/{output_filename}-submission.csv')
    
    return

def get_important_features(
    X,
    y,
    estimator,
    n_iterations=100,
    random_seed=123,
    n_features=100
):
    """
    Function to calculate stable feature importances (FI).
    
    Params:
    -------
    X - Feature space data as pandas dataframe
    y - Target variable
    estimator - must be SKLEARN RandomForestClassifier
    n_iterations - number of models fit over which FI statistics are calculated
    random_seed - The random seed will allow for reproducible results, 
                  given the same n_iterations and random_seed values.
    n_features -
    
    Returns:
    ------- 
    - n_features
    """
    
    if estimator.random_state:
        raise ValueError(
    "Estimators cannot be passed with a random_state parameter. Please use the random_seed parameter."
        )
    
    np.random.seed(random_seed)
    
    importance_df = pd.DataFrame(index=X.columns)

    for n in tqdm(range(n_iterations), total=n_iterations):
        estimator.fit(X, y)
        importance_df[f'{n}'] = estimator.feature_importances_

    importance_df['mean'] = importance_df.iloc[:, :n_iterations].mean(axis=1)

    top_features = list(importance_df['mean'].sort_values(ascending=False).index)

    if type(n_features) == float:
        n = int(len(top_features) * n_features)
        return top_features[:n]
    elif type(n_features) == int:
        return top_features[:n]
