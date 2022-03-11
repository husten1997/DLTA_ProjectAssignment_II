import pandas as pd
import tensorflow as tf

def docPresentation(corpus: pd.Series, vector_size, embedding_matrix):
    '''
    :param corpus: raw data which includes the preprocessed text
    :param vector_size: the vector size which is used for the word embedding
    :param embedding_matrix: the matrix as result of the word embedding
    :return: matrix of the dimension docs x vector_size which has as input the sum of each vector_size dimension of the embedding matrix per document
    '''


    #create list of unigrams in the overall corpus
    lst_corpus = []

    for string in corpus:
        lst_words = string.split()
        lst_grams = [' '.join(lst_words[i:i+1])
                     for i in range(0, len(lst_words), 1)]
        lst_corpus.append(lst_grams)

    #initialize tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(lower = True, split = " ",
                                                      oov_token = 'NaN')
    #fit tokenizer
    tokenizer.fit_on_texts(lst_corpus)

    #create vocab
    vocab = tokenizer.word_index

    doc_presentation = pd.DataFrame(columns = [i for i in range(vector_size)], index = [i for i in range(corpus.shape[0])])


    #compare tokens with words in the word embedding matrix

    for doc in corpus.index:
        sub_corpus = corpus[doc]

        sub_vocab = sub_corpus.split()

        #helper list
        sum_helper = pd.DataFrame(columns = [i for i in range(vector_size)], index = sub_vocab)

        for voc in range(len(sub_vocab)):
            for word in range(len(vocab.items())):
                try:
                    if sub_vocab[voc] == embedding_matrix.index[word][0]:
                        sum_helper.iloc[voc,:] = embedding_matrix.iloc[word,:]
                except:
                    pass


        doc_presentation.iloc[doc,:] = sum_helper.sum()

    return doc_presentation
