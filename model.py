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