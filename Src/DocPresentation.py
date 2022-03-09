import pandas as pd
import numpy as np
from tqdm import tqdm


def docPresentation_alt(corpus: pd.Series, embedding_matrix: pd.DataFrame, method: str = 'sum') -> pd.DataFrame:
    '''
    Function which converts the word embedding into a document representation.

    :param corpus: raw data which includes the preprocessed text
    :param embedding_matrix: the matrix as result of the word embedding
    :return: DataFrame of the dimension docs x vector_size which has as input the sum of each vector_size dimension of the embedding matrix per document
    '''

    vocab = corpus.str.cat(sep = " ").split()
    vocab = np.unique(vocab)

    vector_size = embedding_matrix.shape[1]

    doc_presentation = pd.DataFrame(columns = [i for i in range(vector_size)], index = corpus.index)


    for i, doc in tqdm(zip(corpus.index, corpus)):

        words = doc.split()
        sum_helper = embedding_matrix.loc[words, :]

        #print(sum_helper.sum(axis = 0))
        if method == "sum":
            doc_presentation.loc[i,:] = sum_helper.sum(axis = 0)
        elif method == "average":
            doc_presentation.loc[i, :] = sum_helper.mean(axis=0)

    return doc_presentation