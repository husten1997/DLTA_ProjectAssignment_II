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
<<<<<<< Updated upstream
=======


def dataSelection(data: pd.DataFrame, label_stage: int, doc_type: str = 'question'):
    """
    Function which handles the filtering of the data.

    :param data: DataFrame of the data
    :param label_stage: Integer value of the label stage (range from 1 to 3)
    :param doc_type: Type of the document (range 'question', 'answer')
    :return: returns a DataFrame with the filtered data
    """
    label_stage_str = f"label_l{label_stage}"
    type = {'question': 'QID', 'answer': 'AID'}
    type_label = {'question': 'Question', 'answer': 'Answer'}
    label_str = f"{type_label[doc_type]}_{label_stage}"

    selection_id = data['label_id'].str.contains(type[doc_type])
    selection_label = data[label_stage_str].str.contains(label_str)

    return data.loc[[x&y for x, y in zip(selection_id, selection_label)], :]


def dataSample(data: pd.DataFrame, method: str, n: int, col: str = 'label_l1'):
    """
    Handles the data sampling.

    :param data: dataset
    :param method: method of sampling ('undersample', 'resample')
    :param n: desired length of sampled data (obsolete for undersampling because the number of samples of the smallest group is set as n)
    :param col: name of the target column
    :return: array of indices
    """
    labels = np.unique(data[col])
    label_counts = {}
    for l in labels:
        label_counts[l] = label_counts[l] = data[col].str.cat(sep = " ").split().count(str(l))

    data_groups = {}
    for l in labels:
        data_groups[l] = data.loc[data[col] == l, :].index

    selection_index = []
    if method == "undersample":
        n = min(label_counts.values())
        for l in labels:
            selection_index.extend(np.random.choice(data_groups[l], n))
    elif method == "resample":
        for l in labels:
            selection_index.extend(np.random.choice(data_groups[l], n, replace = True))

    return selection_index


def dataSplit(data: pd.DataFrame, selection_index, train_split_frac: float = 0.5):
    """
    Handls the splitting of data (data and document representation that is, every data as long the row-index matches the doc-index)

    :param data: DataFrame of data
    :param selection_index: selection_index (see dataSample)
    :param train_split_frac: fraction of the data which should be considered for the train dataset
    :return: tuple of train and test data
    """
    data = data.loc[selection_index, :]
    data = data.sample(frac=1)

    split_index = np.floor(train_split_frac * data.shape[0])
    return data.loc[data.index < split_index, :], data.loc[data.index < split_index, :]


def generateEncodingMatrix(data: pd.Series) -> pd.DataFrame:
    """
    Generates a DataFrame which holds a 'index' column (number) and a 'value' column (labels). Each column can be set as index and therefore a translation between labels and number representation is possible.

    :param data: Series of labels
    :return: DataFrame
    """
    labels = np.unique(data)
    encoding_matrix = pd.DataFrame([[i, v] for i, v in enumerate(labels)], columns=['index', 'value'])
    return encoding_matrix


def dataEncoding(data: pd.Series, encoding_matrix: pd.DataFrame):
    """
    Handles the encoding (translation from text labels to number representation) of the labels.

    :param data: Series of (text) labels
    :param encoding_matrix: encoding_matrix (see generateEncodingMatrix)
    :return: array of encoded data
    """
    return encoding_matrix.set_index('value').loc[data, 'index'].values


def dataDecoder(data, encoding_matrix):
    """
    Handles the decoding (translation from number representation labels to text labels) of the labels.

    :param data: array/Series of numbers
    :param encoding_matrix: encoding_matrix (see generateEncodingMatrix)
    :return: array of decoded data
    """
    return encoding_matrix.set_index('index').loc[data, 'value'].values
>>>>>>> Stashed changes
