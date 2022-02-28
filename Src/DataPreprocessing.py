import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import SpaceTokenizer
from nltk.stem import WordNetLemmatizer


# Checks the data for missing values (na) and (optionally) removes them.
#
#   data: data object (pd.DataFrame)
#
#   keepNA: the rows containing nas are not dropped, the function then only returns the na count per column and number
#           or rows containing at least one na
#
#   keepOldIDs: the pd.dropna function drops the rows containing nas, but does not change the indexing, which leads to
#               missing index values. Therefore, for loops iterating over row-indices will not work anymore. Therefore,
#               the index is replaces by a new index. If the old index is import it can be saved as column of
#               the DataFrame

def dataCheck(data, keepNA = False, keepOldIDs = False):
    print(data.isna().sum())

    rowwiseIsNA = np.vectorize(pd.isna)
    rowwiseNA = rowwiseIsNA(data)
    numRemovedRows = sum([any(row) for row in rowwiseNA])
    print(f"There will be {numRemovedRows} rows removed")

    if not keepNA:
        data = data.dropna(axis = 0)
        data = data.reset_index(drop=not keepOldIDs)

    return data


# Wrapper function for the text preprocessing. Applies the following text preprocessing steps:
#   1) removes capitalisation (output column of this step: 'proText_C')
#   2) removes punctuation (output column of this step: 'proText_CP')
#   3) removes stopwords (output column of this step: 'proText_CPS')
#   4) Lemmitazation (stemming) of the tokens (output column of this step: 'proText_CPSL')
#
#   data: data object (pd.DataFrame)
#
#   input_col: name of the column containing the original text
#
#   keepSteps: Boolean-Value which determines if the columns which results after each of the four steps are
#              dropped or kept
#
#   keepOriginalData: Boolean-Value which determines if the original text data is saved or not
#
def textPreprocessing(data, input_col = 'text', keepSteps = True, keepOriginalData = True):
    data = textCapitalisation(data)
    data = textRmPunctuaion(data)
    data = textRmStopwords(data)
    data = textLemmatization(data)

    data[str(str(input_col) + "_original")] = data[input_col]
    data[input_col] = data['proText_CPSL']

    if not keepSteps:
        data = data.drop(['proText_C', 'proText_CP', 'proText_CPS', 'proText_CPSL'], axis = 1)

    if not keepOriginalData:
        data = data.drop([str(str(input_col) + "_original")], axis = 1)

    return data


# Function which covers the removal of the capitalisation. Therefore, the np.vectorize function is used to create a new
# function which returns an array of strings. Each element in this array of strings is created from one element of the
# text column to which the str.lower function is applied. This procedure basically leads to an elementwise application
# of the str.lower function. The results are then saved in a new column.
#
#   data: Data object (pd.DataFrame)
#
#   input_col: Name of the column containing the strings of the previous step.
#
#   output_col: Name of the column which will contain the texts after the application of the current
#               step/transformation.
#
def textCapitalisation(data, input_col = 'text', output_col ='proText_C'):
    # %% Data Preparation - Capitalisation
    rmCap = np.vectorize(str.lower)
    data[output_col] = rmCap(data[input_col])

    return data


# Function which covers the removal of punctuation. Therefore, the punctuation, digits from the strings library and some
# costume strings are used. With the help of the maketrans and translate function each punctuation and digit character
# is removed (or rather replaced by '') in each text. Additionally, double spaces are replaced, which could result from
# the removal of punctuation (e.x. "We spend 100$ on xy." => "We spend  on xy"). Afterwards leading and trailing
# spaces are removed from the texts.
#
#   data: Data object (pd.DataFrame)
#
#   input_col: Name of the column containing the strings of the previous step.
#
#   output_col: Name of the column which will contain the texts after the application of the current
#               step/transformation.
#
def textRmPunctuaion(data, input_col = 'proText_C', output_col ='proText_CP'):
    # %% Data Preparation - Remove leading/trailing spaces and punctuation
    for s in range(data.shape[0]):
        data.loc[s, output_col] = data.loc[s, input_col].translate(
            str.maketrans('', '', string.punctuation + string.digits + "–" + "‘"))  # .replace("'", "")
        data.loc[s, output_col] = str(data.loc[s, output_col]).replace('  ', ' ')

    rmLeadSpace = np.vectorize(str.lstrip)
    data[output_col] = rmLeadSpace(data[output_col])

    return data


# Function which covers the removal of stopwords. For that the nltk packages is used which contains a list
# of common (english) stopwords. To this list were some obviously missing words added. Because of the previous step,
# the stopwords have to be stripped of their punctuation. Then each string element is seperated into its tokens and all
# tokens which are in the list of stopwords are removed. Afterwards a new string is created. The results have shown
# that some special characters where not removed correctly, therefore the punctuation is removed again.
# After that the resulting word list and word frequency's in the corpus where analysed (upper 5% of word frequencies)
# and a list of additional stopwords (with high frequencies) was created, which are then removed separately.
#
#   data: Data object (pd.DataFrame)
#
#   input_col: Name of the column containing the strings of the previous step.
#
#   output_col: Name of the column which will contain the texts after the application of the current
#               step/transformation.
#
def textRmStopwords(data, input_col ='proText_CP', output_col ='proText_CPS'):
    # %% Data Preparation - Removal of stopwords
    nltk.download('stopwords')
    nltk.download('punkt')

    all_stopwords = stopwords.words('english')
    all_stopwords.extend(["i'm", "it'll"])

    all_stopwords_noPunc = []
    for word in all_stopwords:
        all_stopwords_noPunc.append(word.translate(str.maketrans('', '', string.punctuation + string.digits + "–")))

    all_stopwords.extend(all_stopwords_noPunc)

    tk = SpaceTokenizer()

    for s in range(data.shape[0]):
        tokens = tk.tokenize(data.loc[s, input_col])
        tokens_noStop = [word for word in tokens if not word in all_stopwords]
        data.loc[s, output_col] = (" ").join(tokens_noStop)
        data.loc[s, output_col] = data.loc[s, output_col].translate(
            str.maketrans('', '', string.punctuation + string.digits + "’"))
        data.loc[s, output_col] = str(data.loc[s, output_col]).replace('  ', ' ')

    additinoal_stopwords = ["were", "us", "going", "thats", "well", "weve", "one", "also", "q", "–", "still", "its",
                            "g", "theyre", "may", "ill", "id", "dont", "ive", "cant", "theyve", "im", "youre", "hi"]

    for s in range(data.shape[0]):
        tokens = tk.tokenize(data.loc[s, output_col])
        tokens_noStop = [word for word in tokens if not word in additinoal_stopwords]
        data.loc[s, output_col] = (" ").join(tokens_noStop)

    return data

# Function which covers lemmatization. Again the nltk packages is used here. Each element in the text columns is again
# seperated into its tokens and then the stem of the corresponding token is returned and combined back to a string.
#
#   data: Data object (pd.DataFrame)
#
#   input_col: Name of the column containing the strings of the previous step.
#
#   output_col: Name of the column which will contain the texts after the application of the current
#               step/transformation.
#
def textLemmatization(data, input_col ='proText_CPS', output_col ='proText_CPSL'):
    # %% Data Preparation - Lemmatization
    nltk.download('wordnet')
    nltk.download('omw-1.4')


    tk = SpaceTokenizer()
    lemmatizer = WordNetLemmatizer()

    for s in range(data.shape[0]):
        tokens = tk.tokenize(data.loc[s, input_col])
        data.loc[s, output_col] = (" ").join([lemmatizer.lemmatize(w) for w in tokens])

    return data
