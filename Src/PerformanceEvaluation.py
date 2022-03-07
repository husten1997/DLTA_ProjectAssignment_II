#%%
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, plot_roc_curve, roc_auc_score
import matplotlib.pylab as plt


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



def roc_measure(data,doc_type, label_stage, data_pred):

    '''

    :param data: data with labeled text
    :param doc_type: Question or answer
    :param label_stage: stage of the classified labels
    :param data_pred: predictions of a classifier with probabilities as output
    :return: roc table, roc curve and area under the curve per probability
    '''


    data_true = []
    roc_table = pd.DataFrame(columns = ["threshold", "TPR", "FPR", "AUC"], index = [i for i in range(100)])


    for doc in range(data.shape[0]):
        #check if doc type is equal to the category of interest (coi)
        try:
            if data[label_stage][doc].startswith(doc_type) == True:

                data_true.append(data[label_stage][doc])
        except:
            pass


    #binarize the data

    y_coi = []

    for label in range(len(data_true)):
        y = int(data_true[label] == 'Question_1_Company_specific')
        y_coi.append(y)

    #binarize predicted data with probability threshold


    for threshold in range(100):

        y_hat = []

        for label in range(len(data_true)):
            y = int(data_pred[label] >= threshold/100)
            y_hat.append(y)

        fpr, tpr, _ = roc_curve(y_true = y_coi, y_score = y_hat)

        roc_table.loc[threshold] = threshold/100, fpr[1], tpr[1], roc_auc_score(y_true = y_coi, y_score = y_hat)

    plt.figure()
    lw = 2
    plt.plot(
        roc_table["FPR"],
        roc_table["TPR"],
        lw = lw
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    return roc_table




