import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV, train_test_split, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

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

    Prints train, test, cross-val log-loss.
    
    """
    y_train = train.loc[:, target_col]  
    X_train = train.drop([target_col, 'field_id'], axis=1)
    
    y_test = test.loc[:, target_col]
    X_test = test.drop([target_col, 'field_id'], axis=1)

    estimator.fit(X_train, y_train)

    train_pred = estimator.predict_proba(X_train)
    test_pred = estimator.predict_proba(X_test)
    
    train_score = log_loss(y_train, train_pred)
    test_score = log_loss(y_test, test_pred) 
    
    # print('Model evaluation: log-loss')
    # print('--' * 20)
    # print(f'Train score: {train_score}')
    # print(f'Test score: {test_score}')
          
    return test_score

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

    for n in range(n_iterations):
        estimator.fit(X, y)
        importance_df[f'{n}'] = estimator.feature_importances_

    importance_df['mean'] = importance_df.iloc[:, :n_iterations].mean(axis=1)
    # importance_df['std']  = importance_df.iloc[:, :n_iterations].std(axis=1)
    # importance_df['min'] = importance_df.iloc[:, :n_iterations].min(axis=1)
    # importance_df['max'] = importance_df.iloc[:, :n_iterations].max(axis=1)

    top_features = list(importance_df['mean'].sort_values(ascending=False).index)

    if type(n_features) == float:
        n = int(len(top_features) * n_features)
        return top_features[:n]
    elif type(n_features) == int:
        return top_features[:n]



    

    