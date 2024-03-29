import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from Src.prompt import prompt


def classifierLinear(sampleTrain: pd.DataFrame, docRepresentation: pd.DataFrame, data: pd.Series,
                     label_col: str = 'label_l1', prediction_suffix: str = "_LINpred", prob_suffix: str = "_LINprob"):
    """
    Function for the whole classification process by logistic regression.

    :param sampleTrain: pd.DataFrame containing a column for the document representation of the train data and a colum for the labels of the train data
    :param docRepresentation: complete document representation (can be different from fitting data, i.e. all documents can be used)
    :param data: complete dataset, can be different from the fitting data
    :param label_col: name of the target colum
    :param epochs: number of epochs
    :param prediction_suffix: suffix for the prediction colum
    :param prob_suffix: suffix for the probability colum
    :return: dataframe with the original data plus the colum of predicted label data
    """

    prompt("Starting Linear Classifier")
    unlist = lambda x: np.array([np.array(i) for i in np.array(x)])

    docRepresentation = unlist(docRepresentation)
    docRepTrain = unlist(sampleTrain['doc'])

    X_train = np.array(docRepTrain)
    X_test = np.array(docRepresentation)

    y_train = sampleTrain['labels']

    logit = LogisticRegression()
    logit.fit(X_train, y_train)

    pred_data = logit.predict(X_test)
    pred_data_prob = logit.predict_proba(X_test)

    pred_data_prob_marketrel = np.array(pred_data_prob)[:, 1]

    data_ = pd.DataFrame()
    data_[label_col] = data
    data_[str(label_col + prediction_suffix)] = pred_data
    data_[str(label_col + prob_suffix)] = pred_data_prob_marketrel

    return data_
