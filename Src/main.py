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
w2v_embedding = w2v_matrix(corpus = corpus, window_size = 5,min_count = 1, sg = 1,vector_size = 300)

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
roc_measure(data, NN_w2v_pred, encoding_matrix,True,'w2v')

#oversampling example

word_embedding = w2v_doc
labels = data['label_l1']

vectors, labels = oversampling(word_embedding, labels, 'SMOTE')