import pandas as pd
import time

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

    preds_df.to_csv(f'./submissions/{output_filename}.csv')
    
    return