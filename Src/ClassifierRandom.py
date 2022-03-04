import numpy as np
import pandas as pd


def randomClassifier(data: pd.DataFrame, input_col: str = 'label_l1', output_col: str = 'label_l1_pred'):
    pred = np.random.choice(a = np.unique(data[input_col]), size=data.shape[0])
    data[output_col] = pred
    return data
