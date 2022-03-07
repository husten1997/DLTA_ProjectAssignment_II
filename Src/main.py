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
mat_tfidf = tf_idf(corpus)

#%% LSA
doc_mat_lsa = LSA(mat_tfidf)

#%% GloVe
context_matrix, target_matrix = GloVe(corpus, epochs = 40, eta=0.0001)

#%% word2vec
w2v_embedding = w2v_matrix(corpus = corpus, window_size = 5,min_count = 5, sg = 1,vector_size = 300)

#%% doc presentation of w2v
w2v_doc = docPresentation(corpus = corpus, vector_size = 300, embedding_matrix = w2v_embedding)



