#%% Imports
# General Import
import numpy as np
import pandas as pd

# Own Modules Import
from Src.DataImport import DataImport
from Src.DataPreprocessing import textPreprocessing
from Src.DataPreprocessing import dataCheck
from Src.GloVe import GloVe

#%% Data Import
data = DataImport()

#%% Data Check
data = dataCheck(data)

#%% Data Preprocessing
data = textPreprocessing(data)

#%% GloVe
context_matrix, target_matrix = GloVe(data['text'], epochs = 40, eta=0.0001)



