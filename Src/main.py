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
lsa_doc = LSA(mat_tfidf)

#%% GloVe
context_matrix, target_matrix = GloVe(corpus, epochs = 40, eta=0.0001)
glove_doc = docPresentation_alt(corpus = corpus, embedding_matrix = context_matrix + target_matrix)

#%% word2vec
w2v_embedding = w2v_matrix(corpus = corpus, window_size = 5,min_count = 5, sg = 1,vector_size = 300)

#%% doc presentation of w2v
w2v_doc = docPresentation(corpus = corpus, vector_size = 300, embedding_matrix = w2v_embedding)
w2v_doc_alt = docPresentation_alt(corpus = corpus, embedding_matrix = w2v_embedding)
#%% ClassifierNN

# Data Prep
selection_index = dataSample(data, method = "undersample", n = 0)
encoding_matrix = generateEncodingMatrix(data['label_l1'])

dataTrain, dataTest = dataSplit(data, selection_index)

docTrain_tfidf, docTest_tfidf = dataSplit(tfidf_doc, selection_index)
docTrain_glove, docTest_glove = dataSplit(glove_doc, selection_index)

# Classifier Fit
test_data_tfidf = classifierLinear(docTrain_tfidf, tfidf_doc, dataTrain, data)
test_data_glove = classifierNN(docTrain_glove, docTest_glove, glove_doc.loc[selection_index, :], dataTrain, dataTest, data.loc[selection_index, :], encoding_matrix)

# Performance Evaluation
conf_matrix_tfidf = calculateConfusionMatrix(test_data_tfidf['label_l1'], test_data_tfidf['label_l1_LINpred'])
#TODO: roc_measure for linear classifier

conf_matrix_glove = calculateConfusionMatrix(test_data_glove['label_l1'], test_data_glove['label_l1_NNpred'])
roc_measure(dataTrain,'Question_1_Market_related', test_data_glove)



