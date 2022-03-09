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
from Src.DataPreprocessing import  oversampling
from Src.PerformanceEvaluation import calculateConfusionMatrix, roc_measure
from Src.ClassifierLinear import classifierLinear
from Src.ClassifierNN import classifierNN
from tqdm import tqdm
from Src.prompt import prompt

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
context_matrix, target_matrix = GloVe(corpus, epochs = 40, eta=0.0001, overwrite=False)
glove_doc = docPresentation_alt(corpus = corpus, embedding_matrix = context_matrix + target_matrix)

#%% word2vec
w2v_embedding = w2v_matrix(corpus = corpus, window_size = 5,min_count = 1, sg = 1,vector_size = 300)
w2v_doc = docPresentation_alt(corpus = corpus, embedding_matrix = w2v_embedding)

"""
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
roc_measure(data, NN_w2v_pred, encoding_matrix,True,'w2v')

#oversampling example

word_embedding = w2v_doc
labels = data['label_l1']

vectors, labels = oversampling(word_embedding, labels, 'SMOTE')
"""


#%% Prep Classifier

# Data Prep
sample_method = "SMOTE"
nfolds = 10
encoding_matrix = generateEncodingMatrix(data['label_l1'])

tfidf_sample = dataSample(tfidf_doc, data, method = sample_method, n = 0, folds=nfolds)
lsa_sample = dataSample(lsa_doc, data, method = sample_method, n = 0, folds=nfolds)
glove_sample = dataSample(glove_doc, data, method = sample_method, n = 0, folds=nfolds)
w2v_sample = dataSample(w2v_doc, data, method = sample_method, n = 0, folds=nfolds)

# Sample Split:
convert_to_dic = lambda x: {l:i for l, i in zip(['dataTrain', 'dataTest'], x)}

tfidf_data = convert_to_dic(dataSplit(tfidf_sample))
lsa_data = convert_to_dic(dataSplit(lsa_sample))
glove_data = convert_to_dic(dataSplit(glove_sample))
w2v_data = convert_to_dic(dataSplit(w2v_sample))

#%% NN Classifier
# Fitt classifier
data_tfidf_nn = classifierNN(tfidf_data['dataTrain'], tfidf_data['dataTest'], tfidf_doc, data, encoding_matrix, project_name="NN_tfidf", plot_title="TfIdf - NN Classifier Loss")
data_lsa_nn = classifierNN(lsa_data['dataTrain'], lsa_data['dataTest'], lsa_doc, data, encoding_matrix, project_name="NN_lsa", plot_title="LSA - NN Classifier Loss")
data_glove_nn = classifierNN(glove_data['dataTrain'], glove_data['dataTest'], glove_doc, data, encoding_matrix, project_name="NN_glove", plot_title="GloVe - NN Classifier Loss")
data_w2v_nn = classifierNN(w2v_data['dataTrain'], w2v_data['dataTest'], w2v_doc, data, encoding_matrix, project_name="NN_w2v", plot_title="W2V - NN Classifier Loss")

# Performance Evaluation
conf_matrix_tfidf = calculateConfusionMatrix(data_tfidf_nn['label_l1'], data_tfidf_nn['label_l1_NNpred'])
auc_tfidf = roc_measure(data_tfidf_nn, encoding_matrix, label_true = 'label_l1', label_prob = 'label_l1_NNprob', plot_title = 'TfIdf - ROC Curve')

conf_matrix_lsa = calculateConfusionMatrix(data_lsa_nn['label_l1'], data_lsa_nn['label_l1_NNpred'])
auc_lsa = roc_measure(data_lsa_nn, encoding_matrix, label_true = 'label_l1', label_prob = 'label_l1_NNprob', plot_title = 'LSA - ROC Curve')

conf_matrix_glove = calculateConfusionMatrix(data_glove_nn['label_l1'], data_glove_nn['label_l1_NNpred'])
auc_glove = roc_measure(data_glove_nn, encoding_matrix, label_true = 'label_l1', label_prob = 'label_l1_NNprob', plot_title = 'GloVe - ROC Curve')

conf_matrix_w2v = calculateConfusionMatrix(data_w2v_nn['label_l1'], data_w2v_nn['label_l1_NNpred'])
auc_w2v = roc_measure(data_w2v_nn, encoding_matrix, label_true = 'label_l1', label_prob = 'label_l1_NNprob', plot_title = 'Word to Vec - ROC Curve')

#%% Linear Classifier

#%% Crossvalidation
resultMatrix = pd.DataFrame(0, columns=['TfIdf', 'LSA', 'GloVe', 'Word to Vec'], index = ['Linear-Classifier', 'Neural-Network-Classifier'])
prompt("Starting Cross-Validation")
for i in tqdm(range(1, nfolds+1)):
    # Sample Split:
    tfidf_data = convert_to_dic(dataSplit(tfidf_sample, test_fold=i))
    lsa_data = convert_to_dic(dataSplit(lsa_sample, test_fold=i))
    glove_data = convert_to_dic(dataSplit(glove_sample, test_fold=i))
    w2v_data = convert_to_dic(dataSplit(w2v_sample, test_fold=i))

    # Linear Classifier
    # Fitt classifier

    # Performance Eval



    # NN Classifier
    # Fitt classifier
    data_tfidf_nn = classifierNN(tfidf_data['dataTrain'], tfidf_data['dataTest'], tfidf_doc, data, encoding_matrix,
                                 project_name="NN_tfidf", plot_title="TfIdf - NN Classifier Loss")
    data_lsa_nn = classifierNN(lsa_data['dataTrain'], lsa_data['dataTest'], lsa_doc, data, encoding_matrix,
                               project_name="NN_lsa", plot_title="LSA - NN Classifier Loss")
    data_glove_nn = classifierNN(glove_data['dataTrain'], glove_data['dataTest'], glove_doc, data, encoding_matrix,
                                 project_name="NN_glove", plot_title="GloVe - NN Classifier Loss")
    data_w2v_nn = classifierNN(w2v_data['dataTrain'], w2v_data['dataTest'], w2v_doc, data, encoding_matrix,
                               project_name="NN_w2v", plot_title="W2V - NN Classifier Loss")

    # Performance Eval
    resultMatrix.loc['Neural-Network-Classifier', 'TfIdf'] += roc_measure(data_tfidf_nn, encoding_matrix, label_true='label_l1', label_prob='label_l1_NNprob',
                            plot_title='TfIdf - ROC Curve', plot=False)
    resultMatrix.loc['Neural-Network-Classifier', 'LSA'] += roc_measure(data_lsa_nn, encoding_matrix, label_true='label_l1', label_prob='label_l1_NNprob',
                          plot_title='LSA - ROC Curve', plot=False)
    resultMatrix.loc['Neural-Network-Classifier', 'GloVe'] += roc_measure(data_glove_nn, encoding_matrix, label_true='label_l1', label_prob='label_l1_NNprob',
                            plot_title='GloVe - ROC Curve', plot=False)
    resultMatrix.loc['Neural-Network-Classifier', 'Word to Vec'] += roc_measure(data_w2v_nn, encoding_matrix, label_true='label_l1', label_prob='label_l1_NNprob',
                          plot_title='Word to Vec - ROC Curve', plot=False)

resultMatrix = resultMatrix / nfolds

print(resultMatrix)