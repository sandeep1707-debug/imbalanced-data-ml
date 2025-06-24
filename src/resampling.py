from sklearn.utils import resample
import pandas as pd

def oversample_minority(data):
    majority = data[data.claim_status == 0]
    minority = data[data.claim_status == 1]
    minority_oversampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    oversampled_data = pd.concat([majority, minority_oversampled])
    return oversampled_data
