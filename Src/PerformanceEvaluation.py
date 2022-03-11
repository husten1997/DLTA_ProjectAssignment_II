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



def roc_measure(data_true,doc_type, prop_pred, plot: bool, plot_title: str):

    '''
    :param data_true: data which includes the label column 'label_l1'
    :param doc_type: Question or Answer Label which shall be classified as 1
    :param prop_pred: predictions of a classifier with probabilities as output
    :return: roc table, roc curve and area under the curve per probability
    '''

    roc_table = pd.DataFrame(columns = ['threshold', 'TPR', 'FPR'], index = [i for i in range(100)])

    y_coi = []

    for label in range(data_true.shape[0]):
        y = int(data_true['label_l1'].iloc[label] == doc_type)
        y_coi.append(y)


    #calculate measures
    fpr, tpr, _ = roc_curve(y_true = y_coi, y_score = prop_pred)
    area_under_curve = roc_auc_score(y_coi, prop_pred)


    if plot == True:
        
        #plot ROC curve
        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            lw = lw)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{plot_title}')
        plt.plot([0, max(fpr)], [0, max(tpr)], color="black", lw=lw, linestyle="--", label = "Random classifier")
        plt.legend()
        plt.show()

    return area_under_curve




