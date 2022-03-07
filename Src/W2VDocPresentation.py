import pandas as pd

def docPresentation(data, vector_size, embedding_matrix):
    '''
    :param data: raw data which includes the preprocessed text
    :param vector_size: the vector size which is used for the word embedding
    :param embedding_matrix: the matrix as result of the word embedding
    :return: matrix of the dimension docs x vector_size which has as input the sum of each vector_size dimension of the embedding matrix per document
    '''

    corpus = data['text']

    doc_presentation = pd.DataFrame(columns = [i for i in range(vector_size)], index = [i for i in range(corpus.shape[0])])

    for corpora in range(data.shape[0]):
        sub_corpus = corpus[corpora]

        sub_vocab = sub_corpus.split()

        #helper list
        sum_helper = pd.DataFrame(columns = [i for i in range(vector_size)], index = sub_vocab)

        for voc in range(len(sub_vocab)):
            for word in range(len(vocab.items())):
                try:
                    if sub_vocab[voc] ==  embedding_matrix.index[word][0]:
                        sum_helper.iloc[voc,:] = embedding_matrix.iloc[word,:]
                except:
                    pass


        doc_presentation.iloc[corpora,:] = sum_helper.sum()
    return doc_presentation
