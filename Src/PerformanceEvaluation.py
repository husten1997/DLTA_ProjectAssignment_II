#%%
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
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



def roc_measure(data, encoding_matrix, label_true, label_prob, plot_title, plot = True):

    '''
    :param data_true: data which includes the label column 'label_l1'
    :param doc_type: Question or Answer Label which shall be classified as 1
    :param prop_pred: predictions of a classifier with probabilities as output
    :return: roc table, roc curve and area under the curve per probability
    '''
    data_true = data[label_true]
    prop_pred = data[label_prob]
    roc_table = pd.DataFrame(columns = ['threshold', 'TPR', 'FPR'], index = [i for i in range(100)])

    #binarize the data
    y_coi = []

    for label in range(data_true.shape[0]):
        y = int(data_true.iloc[label] == encoding_matrix.set_index('index').loc[1, 'value'])
        y_coi.append(y)

    #binarize predicted data with probability threshold
    for threshold in range(101):

        y_hat = []

        for label in range(len(prop_pred)):
            y = int(prop_pred.iloc[label] >= threshold/100)
            y_hat.append(y)

        fpr, tpr, _ = roc_curve(y_true = y_coi, y_score = y_hat)

        roc_table.loc[threshold] = threshold/100, fpr[1], tpr[1]
        area_under_curve = roc_auc_score(y_coi, prop_pred)

    if plot == True:
        #plot ROC curve
        plt.figure()
        lw = 2
        plt.plot(
            roc_table['FPR'],
            roc_table['TPR'],
            lw = lw
        )
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{plot_title}')
        plt.show()

    return area_under_curve




