import numpy as np
import pandas as pd


def randomClassifier(data: pd.DataFrame, input_col: str = 'label_l1', output_col: str = 'label_l1_pred'):
    """
    Random classifier, used for testing. It just randomly assigns labels to the row in the dataset.

    :param data: pd.DataFrame
    :param input_col: Name of the column with the true labels
    :param output_col: Name of the column with the predicted labels
    :type data: object
    """
    pred = np.random.choice(a = np.unique(data[input_col]), size=data.shape[0])
    data[output_col] = pred
    return data
