#%% Imports
# General Import
import numpy as np
import pandas as pd

# Own Modules Import
from Src.DataImport import DataImport
from Src.DataPreprocessing import textPreprocessing
from Src.DataPreprocessing import dataCheck

#%% Data Import
data = DataImport()

#%% Data Check
data = dataCheck(data)

#%% Data Preprocessing
data = textPreprocessing(data)

<<<<<<< Updated upstream
=======
#%%
data = dataSelection(data, label_stage=1, doc_type='question')

#%%
corpus = data['text']

#%% td-idf
mat_tfidf = tf_idf(corpus)

#%% LSA
doc_mat_lsa = LSA(mat_tfidf)

#%% GloVe
context_matrix, target_matrix = GloVe(corpus, epochs = 40, eta=0.0001)
glove_doc = docPresentation_alt(corpus = corpus, embedding_matrix = context_matrix + target_matrix)

#%% word2vec
w2v_embedding = w2v_matrix(corpus = corpus, window_size = 5,min_count = 5, sg = 1,vector_size = 300)

#%% doc presentation of w2v
w2v_doc = docPresentation(corpus = corpus, vector_size = 300, embedding_matrix = w2v_embedding)
w2v_doc_alt = docPresentation_alt(corpus=corpus, embedding_matrix=w2v_embedding)
#%% ClassifierNN

# Data Prep
selection_index = dataSample(data, method = "undersample", n = 0)
encoding_matrix = generateEncodingMatrix(data['label_l1'])

dataTrain, dataTest = dataSplit(data, selection_index)
docTrain, docTest = dataSplit(glove_doc, selection_index)

# Classifier Fit
test_data = classifierNN(docTrain, docTest, glove_doc.loc[selection_index, :], dataTrain, dataTest, data.loc[selection_index, :], encoding_matrix)

# Performance Evaluation
conf_matrix = calculateConfusionMatrix(test_data['label_l1'], test_data['label_l1_NNpred'])
#roc_measure(test_data, 'Q', 'label_l1', test_data)

>>>>>>> Stashed changes

