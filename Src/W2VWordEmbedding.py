import gensim
import pandas as pd
import string
import tensorflow as tf
import numpy as np

###########################
## Tensor representation ##
###########################

def w2v_tensor(window_size, min_count, sg, vector_size):

    '''
    :param data: data which includes the preprocessed text
    :param window_size: window_size which is used for the w2c model
    :param min_count: minimum of count to be counted as
    :param sg: 1 = skip-gram, 0 = CBOW
    :param vector_size: vector_size which is used for the word embedding
    :return: word embedding matrix per document, represented as a tensor
    '''

    doc_tensor = []

    for doc in range(corpus.shape[0]):
        #create unigrams
        lst_corpus = corpus.split()

        '''
        ## detect bigrams and trigrams
        bigrams_detector = gensim.models.phrases.Phrases(lst_corpus,
                                                         delimiter=" ".encode(), min_count=min_count, threshold=10)
        trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus],
                                                          delimiter=" ".encode(), min_count=min_count, threshold=10)
        
        bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
        trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)
        
        lst_corpus = list(bigrams_detector[lst_corpus])
        lst_corpus = list(trigrams_detector[lst_corpus])
        '''

        #initialize tokenizer
        tokenizer = tf.keras.preprocessing.text.Tokenizer(lower = True, split = " ",
                                                          oov_token = 'NaN')

        #fit tokenizer
        tokenizer.fit_on_texts(lst_corpus)

        #create sub vocab
        sub_vocab = tokenizer.word_index

        #create raw version of final matrix
        embedding_matrix = pd.DataFrame(columns = [i for i in range(vector_size)], index = vocab.keys())

        #attach embedding_matrix to doc tensor

        doc_tensor.append(embedding_matrix)


        #initialize model
        w2v = gensim.models.word2vec.Word2Vec(sub_vocab.items(),
                                              window=window_size, min_count=min_count, sg=sg,vector_size=vector_size)

        for token,doc in sub_vocab.items():
            try:
                doc_tensor[doc] = w2v[token]
            except:
                pass

    return doc_tensor


def w2v_matrix(corpus: pd.Series, window_size: int, min_count: int, sg: int, vector_size: int):
    '''
    Function which handles the fitting of the word-to-vec algorithm.

    :param data: data which includes the preprocessed text
    :param window_size: window_size which is used for the w2c model
    :param min_count: minimum of count to be counted as
    :param sg: 1 = skip-gram, 0 = CBOW
    :param vector_size: vector_size which is used for the word embedding
    :return: word embedding matrix per document, represented as a matrix
    '''

    #create unigrams
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

    #create raw version of final matrix
    embedding_matrix = pd.DataFrame(columns = [i for i in range(vector_size)], index = vocab.keys())

    #initialize model
    w2v = gensim.models.word2vec.Word2Vec(lst_corpus,
                                          window=window_size, min_count=min_count, sg=sg,vector_size=vector_size)

    for token in vocab.keys():
        try:
            #prompt(f"calculate word embedding for the token {token}")
            embedding_matrix.loc[token] = w2v.wv[token]
        except:
            pass

    return embedding_matrix