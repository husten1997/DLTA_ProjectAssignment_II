import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def classifierLinear(docRepTrain: pd.DataFrame, docRepresentation: pd.DataFrame, dataTrain: pd.DataFrame,
                     data: pd.DataFrame, label_col: str ='label_l1', prediction_suffix: str = "_LINpred"):

    X_train = np.matrix(docRepTrain)
    X_test = np.matrix(docRepresentation)

    y_train = dataTrain[label_col]

    logit = LogisticRegression()
    logit.fit(X_train, y_train)

    pred_data = logit.predict(X_test)

    data[str(label_col + prediction_suffix)] = pred_data

    return data
