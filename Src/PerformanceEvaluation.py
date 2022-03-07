#%%
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, plot_roc_curve


def calculateConfusionMatrix(data_true: pd.Series, data_pred: pd.Series, freq: bool = True):
    """
    Wrapper function for sklearn.metrics.confusion_matrix. It can additionally handle frequency representation of the confusion matrix.

    :param data_true: Series of the true data
    :param data_pred: Series of predicted data
    :param freq: Boolean which determines if the returned confusion matrix should be in absolut values or frequency representation
    :return: DataFrame of confusion matrix
    """
    labels = np.unique(data_true)
    con = confusion_matrix(data_true, data_pred, labels=labels)
    if freq:
        con = con / con.flatten().sum()

    con = pd.DataFrame(con, columns=["pred " + x for x in labels], index=labels)
    return con


def plotROC(data_true: pd.Series, data_pred: pd.Series):
    None



def roc_measure(class_dimension, doc_type, label_stage, y_hat):

    y_coi = []

    class_dimension = 1

    doc_type = "Answer"
    label_stage = "label_l1"

    for doc in range(data.shape[0]):
        #check if doc type is equal to the category of interest (coi)
        try:
            if data[label_stage][doc].startswith(doc_type) == True:

                y_coi.append(data[label_stage][doc])
        except:
            pass


