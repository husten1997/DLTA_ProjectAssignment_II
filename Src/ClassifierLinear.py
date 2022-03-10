import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def classifierLinear(sampleTrain: pd.DataFrame, docRepresentation: pd.DataFrame, data: pd.DataFrame,
                     label_col: str ='label_l1', prediction_suffix: str = "_LINpred", prob_suffix: str = "_LINprob"):

    docRepTrain = sampleTrain['doc']
    docRepTrain = np.matrix([np.array(i) for i in docRepTrain])


    X_train = np.matrix(docRepTrain)
    X_test = np.matrix(docRepresentation)

    y_train = sampleTrain['labels']

    logit = LogisticRegression()
    logit.fit(X_train, y_train)

    pred_data = logit.predict(X_test)
    pred_data_prob = logit.predict_proba(X_test)

    pred_data_prob_marketrel = np.array(pred_data_prob)[:, 1]

    data[str(label_col + prediction_suffix)] = pred_data
    data[str(label_col + prob_suffix)] = pred_data_prob_marketrel

    return data
