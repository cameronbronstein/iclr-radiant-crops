import pandas as pd
from sklearn.model_selection import ShuffleSplit

def get_shuffle_splits(
    data,
    n_splits=5,
    test_size=0.2,
    random_seed=123
):
    """
    Get stratified train, test sets for cross validation.

    Params
    ------
    - data
    - split_on: default 'field_id' to aggregate rows by field; 
                all field pixels in either train or test to avoid data leakage.
    - n_splits: number of train, test pairs to return.
    - random_seed: maintains reproducibiltiy of sets across different model iterations.
    
    Returns
    -------
    - n_splits train, test sets for cross validation
    """

    ss = ShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_seed
    )

    splits = ss.split(
        data
    )

    return splits

def get_bootstrap(
    train_data, 
    target_col='label', 
    random_seed=123
):
    """
    Oversample minority classes to majority sample frequency.
    This happens after train-test split to preserve representative validation sample

    Params
    -----
    - train_data:
    - target_col
    - random_seed

    Returns
    -------
    Bootstrapped data
    """
    classes = train_data[target_col].unique()
    most_abundant = train_data[target_col].value_counts().values[0]

    samples = [train_data.loc[(train_data[target_col] == label), :].sample(
        most_abundant, 
        replace=True, 
        random_state=random_seed
        ) for label in classes]

    return pd.concat(samples, ignore_index=True)