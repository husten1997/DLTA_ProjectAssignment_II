#%% Imports
# General Import
import numpy as np
import pandas as pd

# Own Modules Import
from Src.DataImport import DataImport
from Src.DataPreprocessing import textPreprocessing
from Src.DataPreprocessing import dataCheck
from Src.DataPreprocessing import dataSelection
from Src.tf_idf import tf_idf
from Src.LSA import LSA
from Src.GloVe import GloVe
from Src.W2VWordEmbedding import w2v_matrix
from Src.W2VDocPresentation import docPresentation
from Src.DocPresentation import docPresentation_alt
from Src.DataPreprocessing import dataSample, dataSplit, generateEncodingMatrix
from Src.PerformanceEvaluation import calculateConfusionMatrix, roc_measure
from Src.ClassifierLinear import classifierLinear
from Src.ClassifierNN import classifierNN

#%% Data Import
data = DataImport()

#%% Data Check
data = dataCheck(data)

#%% Data Preprocessing
data = textPreprocessing(data)

#%%
data = dataSelection(data, label_stage=1, doc_type='question')

#%%
corpus = data['text']

#%% td-idf
mat_tfidf, tfidf_doc = tf_idf(corpus)

#%% LSA
lsa_doc = LSA(corpus, mat_tfidf)

#%% GloVe
context_matrix, target_matrix = GloVe(corpus, epochs = 40, eta=0.0001)
glove_doc = docPresentation_alt(corpus = corpus, embedding_matrix = context_matrix + target_matrix)

#%% word2vec
w2v_embedding = w2v_matrix(corpus = corpus, window_size = 5,min_count = 5, sg = 1,vector_size = 300)

#%% doc presentation of w2v
w2v_doc = docPresentation_alt(corpus = corpus, embedding_matrix = w2v_embedding)

#%% Classifier

# Data Prep
selection_index = dataSample(data, method = "undersample", n = 0)
encoding_matrix = generateEncodingMatrix(data['label_l1'])

dataTrain, dataTest = dataSplit(data, selection_index)

docTrain_tfidf, docTest_tfidf = dataSplit(tfidf_doc, selection_index)
docTrain_lsa, docTest_lsa = dataSplit(lsa_doc, selection_index)
docTrain_glove, docTest_glove = dataSplit(glove_doc, selection_index)
docTrain_w2v, docTest_w2v = dataSplit(w2v_doc, selection_index)


# Classifier Fit
# Linear Classifier
test_data_tfidf_lin = classifierLinear(docTrain_tfidf, tfidf_doc, dataTrain, data)
test_data_lsa_lin = classifierLinear(docTrain_lsa, lsa_doc, dataTrain, data)
# NN Classifier
test_data_glove_nn = classifierNN(docTrain_glove, docTest_glove, glove_doc.loc[selection_index, :], dataTrain, dataTest, data.loc[selection_index, :], encoding_matrix)
test_data_w2v_nn = classifierNN(docTrain_w2v, docTest_w2v, w2v_doc, dataTrain, dataTest, data, encoding_matrix)

#TODO: AN: NN classifier for tdidf and lsa

# Performance Evaluation
# Linear Classifier
conf_matrix_tfidf_lin = calculateConfusionMatrix(test_data_tfidf_lin['label_l1'], test_data_tfidf_lin['label_l1_LINpred'])
conf_matrix_lsa_lin = calculateConfusionMatrix(test_data_lsa_lin['label_l1'], test_data_lsa_lin['label_l1_LINpred'])

# NN Classifier
conf_matrix_glove = calculateConfusionMatrix(test_data_glove_nn['label_l1'], test_data_glove_nn['label_l1_NNpred'])
#TODO: AN: confusion matrix for tdidf and lsa


#roc test

NN_w2v_pred = test_data_w2v_nn['label_l1_NNprob']
roc_measure(data, NN_w2v_pred, encoding_matrix,True, 'w2v')

from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pylab as plt

def roc_measure(data_true,prop_pred, encoding_matrix, plot, plot_title):

    '''
    :param data_true: data which includes the label column 'label_l1'
    :param doc_type: Question or Answer Label which shall be classified as 1
    :param prop_pred: predictions of a classifier with probabilities as output
    :return: roc table, roc curve and area under the curve per probability
    '''

    roc_table = pd.DataFrame(columns = ['threshold', 'TPR', 'FPR'], index = [i for i in range(100)])

    #binarize the data
    y_coi = []

    for label in range(data.shape[0]):
        y = int(data['label_l1'].iloc[label] == encoding_matrix.set_index('index').loc[1, 'value'])
        y_coi.append(y)

    #binarize predicted data with probability threshold
    for threshold in range(101):

        y_hat = []

        for label in range(len(NN_w2v_pred)):
            y = int(NN_w2v_pred.iloc[label] >= threshold/100)
            y_hat.append(y)

        fpr, tpr, _ = roc_curve(y_true = y_coi, y_score = y_hat)

        roc_table.loc[threshold] = threshold/100, fpr[1], tpr[1]
        area_under_curve = roc_auc_score(y_coi, NN_w2v_pred)

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

    return area_under_curve
