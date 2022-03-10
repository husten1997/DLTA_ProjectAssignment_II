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
from Src.DataPreprocessing import oversampling
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

#%% tf-idf
mat_tfidf, tfidf_doc = tf_idf(corpus)

#%% LSA
lsa_doc = LSA(corpus, mat_tfidf)

#%% GloVe
context_matrix, target_matrix = GloVe(corpus, epochs = 40, eta=0.0001, overwrite=False)
glove_doc = docPresentation_alt(corpus = corpus, embedding_matrix = context_matrix + target_matrix)

#%% word2vec
w2v_embedding = w2v_matrix(corpus = corpus, window_size = 5,min_count = 1, sg = 1,vector_size = 300)
w2v_doc = docPresentation_alt(corpus = corpus, embedding_matrix = w2v_embedding)

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
# Fit classifier
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
# Fit classifier
data_tfidf_lin = classifierLinear(tfidf_data['dataTrain'], tfidf_doc, data)
data_lsa_lin = classifierLinear(lsa_data['dataTrain'], lsa_doc, data)
data_glove_lin = classifierLinear(glove_data['dataTrain'], glove_doc, data)
data_w2v_lin = classifierLinear(w2v_data['dataTrain'], w2v_doc, data)

# Performance Evaluation
conf_matrix_tfidf = calculateConfusionMatrix(data_tfidf_lin['label_l1'], data_tfidf_lin['label_l1_LINpred'])
auc_tfidf = roc_measure(data_tfidf_lin, encoding_matrix, label_true = 'label_l1', label_prob = 'label_l1_LINprob', plot_title = 'TfIdf - ROC Curve')

conf_matrix_lsa = calculateConfusionMatrix(data_lsa_lin['label_l1'], data_lsa_lin['label_l1_LINpred'])
auc_lsa = roc_measure(data_lsa_lin, encoding_matrix, label_true = 'label_l1', label_prob = 'label_l1_LINprob', plot_title = 'LSA - ROC Curve')

conf_matrix_glove = calculateConfusionMatrix(data_glove_lin['label_l1'], data_glove_lin['label_l1_LINpred'])
auc_glove = roc_measure(data_glove_lin, encoding_matrix, label_true = 'label_l1', label_prob = 'label_l1_LINprob', plot_title = 'GloVe - ROC Curve')

conf_matrix_w2v = calculateConfusionMatrix(data_w2v_lin['label_l1'], data_w2v_lin['label_l1_LINpred'])
auc_w2v = roc_measure(data_w2v_lin, encoding_matrix, label_true = 'label_l1', label_prob = 'label_l1_LINprob', plot_title = 'Word to Vec - ROC Curve')

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
    # Fit classifier
    data_tfidf_lin = classifierLinear(tfidf_data['dataTrain'], tfidf_doc, data)
    data_lsa_lin = classifierLinear(lsa_data['dataTrain'], lsa_doc, data)
    data_glove_lin = classifierLinear(glove_data['dataTrain'], glove_doc, data)
    data_w2v_lin = classifierLinear(w2v_data['dataTrain'], w2v_doc, data)

    # Performance Eval
    resultMatrix.loc['Neural-Network-Classifier', 'TfIdf'] += roc_measure(data_tfidf_lin, encoding_matrix, label_true='label_l1', label_prob='label_l1_LINprob',
                            plot_title='TfIdf - ROC Curve', plot=False)
    resultMatrix.loc['Neural-Network-Classifier', 'LSA'] += roc_measure(data_lsa_lin, encoding_matrix, label_true='label_l1', label_prob='label_l1_LINprob',
                          plot_title='LSA - ROC Curve', plot=False)
    resultMatrix.loc['Neural-Network-Classifier', 'GloVe'] += roc_measure(data_glove_lin, encoding_matrix, label_true='label_l1', label_prob='label_l1_LINprob',
                            plot_title='GloVe - ROC Curve', plot=False)
    resultMatrix.loc['Neural-Network-Classifier', 'Word to Vec'] += roc_measure(data_w2v_lin, encoding_matrix, label_true='label_l1', label_prob='label_l1_LINprob',
                          plot_title='Word to Vec - ROC Curve', plot=False)

    # NN Classifier
    # Fit classifier
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