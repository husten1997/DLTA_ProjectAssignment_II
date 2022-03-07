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



y_coi = pd.DataFrame(data = y_coi)
y_coi.dropna(inplace = True)
y_coi = y_coi[0:100]

data_pred = np.random.uniform(0,1,len(y_coi))
data_pred = list(data_pred)



def roc_measure(class_dimension, doc_type, label_stage, data_pred, threshold):


    data_true = []
    roc_table = pd.DataFrame(columns = ["threshold", "TPR", "FPR"], index = [i for i in range(100)])

    class_dimension = 1

    doc_type = 'Answer'
    label_stage = 'label_l1'

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
        y = int(data_true[label] == 'Answer_1_specific')
        y_coi.append(y)

    y_hat = []

    for label in range(len(data_true)):
        y = int(data_pred[label] >= threshold)
        y_hat.append(y)


        fpr, tpr, _ = roc_curve(y_true = y_coi, y_score = y_hat)

    roc_table.loc[0] = threshold, fpr[1], tpr[1]



plt.figure()
lw = 2
plt.plot(
    fpr[2],
    tpr[2],
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc[2],
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

