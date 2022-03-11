#%%
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pylab as plt


def calculateConfusionMatrix(data_true: pd.Series, data_pred: pd.Series, freq: bool = True) -> pd.DataFrame:
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


def roc_measure(data: pd.DataFrame, encoding_matrix: pd.DataFrame, label_true: str, label_prob: str, plot_title: str, plot: bool = True):
    '''
    Function which handles the plot of the ROC curve as well as calculating and returning the area under the curve.

    :param data: pd.DataFrame with all the data, but has to contain on column with the true labels and one column with the predicted probabilities
    :param encoding_matrix: encoding matrix
    :param label_true: Name of the column which contains the true labels
    :param label_prob: Name of the colum which contains the predicted probabilities
    :param plot_title: Title of the ROC plot
    :param plot: boolean if a plot should be generated or not
    :return: area under the curve per probability
    '''
    data_true = data[label_true]
    prop_pred = data[label_prob]

    #binarize the data
    y_coi = []

    for label in range(data_true.shape[0]):
        y = int(data_true.iloc[label] == encoding_matrix.set_index('index').loc[1, 'value'])
        y_coi.append(y)

    #calculate ROC and AUC measures
    fpr, tpr, _ = roc_curve(y_true = y_coi, y_score = prop_pred)
    area_under_curve = roc_auc_score(y_coi, prop_pred)

    if plot:
        #plot ROC curve
        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            lw = lw
        )
        plt.plot([0, max(fpr)], [0, max(tpr)], color="black", lw=lw, linestyle="--")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{plot_title}')
        plt.show()

    return area_under_curve




