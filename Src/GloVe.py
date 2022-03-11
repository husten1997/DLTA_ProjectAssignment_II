import numpy as np
import itertools
from collections.abc import Iterable
import pandas as pd
from tqdm import tqdm
from os.path import exists
import matplotlib.pylab as plt
from Src.prompt import prompt


def generateContextDic(corpus: Iterable, window: int = 5) -> dict:
    """
    Generates a dictionary which holds an array with all context words of every word in the vocabulary

    :param corpus: List or array (or pd.Series) of documents (text entries)
    :param window: The context window size (symmetrically around i)
    :return: Dictionary of word (key) and context-word-arrays (values)
    """
    context_dic = {}
    vocab = pd.Series(corpus).str.cat(sep = " ").split()
    vocab = np.unique(vocab)

    for word in vocab:
        context_dic.setdefault(word, [])

    for entry in corpus:
        tokens = [np.NAN for i in range(window)] + entry.split() + [np.NAN for i in range(window)]
        for i in range(window, len(tokens)-window):
            scope = tokens[(i-window):(i+window+1)]
            context_dic[scope.pop(window)].extend(scope)

    return context_dic


def generateCoOccurrenceMatrix(context_dic: dict):
    """
    Function which generates the Co-Occurence-Matrix. It takes a context dictionary as input and then counts the words
    in the context dictionary for each key-value pair.
    Additionally, saves the resulting matrix as csv.

    :param context_dic: Dictionary with the word context-array pairs.
    :return: Matrix of the co-occurences
    """
    vocab = context_dic.keys()
    matrix = pd.DataFrame(np.zeros((len(vocab), len(vocab)), np.float64), index=vocab, columns=vocab)

    for word in tqdm(vocab):
        scope = pd.Series(context_dic[word]).dropna().str.cat(sep = " ").split()
        unique_terms = np.unique(scope)
        for term in unique_terms:
            matrix.loc[word, term] = scope.count(term)
    matrix.to_csv("Data/GloVe_CoOcMatrix.csv")
    return matrix


def GloVe(corpus: Iterable, overwrite: bool = False, eta: float = 0.00001, eta_bias: float = 0.00002, epochs: int = 20, set_seed: int = None, use_weights: bool = False, x_max = 100, alpha = 3/4, word_embedding_dim = 300):
    """
    Function which handles the GloVe fitting.

    :param corpus: Iterable/list/array or pd.Series which holds the document-texts
    :param overwrite: Boolean which determines if the co-occurrence matrix is re-calculated and the old csv is overwritten (takes more time). Otherwise the co-occurance matrix is loeaded from the csv file (if it exists)
    :param eta: learning rate for the gradient-updates of the word-embedding matrix's
    :param eta_bias: learning rate for the gradient-updates of the bias vectors
    :param epochs: number of fitting epochs
    :param set_seed: (optional) seed, if None (default-value) then no seed is used
    :param use_weights: Determines if the weights of the paper (see Jeffrey Pennington, Richard Socher, Christopher D. Manning: GloVe: Global Vectors for Word Representation) should be used.
    :param x_max: cut-off point of the co-occurrences. Default is 100 as proposed in the paper
    :param alpha: default value is 3/4 as proposed in the paper
    :param word_embedding_dim: dimension of the word-embedding vectors
    :return: tuple of context-word-embedding and target-word-embedding matrix
    """
    if not exists("Data/GloVe_CoOcMatrix.csv") or overwrite:
        prompt("Starting generation of Co-Occurrence Matrix")
        cooc_matrix = generateCoOccurrenceMatrix(generateContextDic(corpus))
    else:
        prompt("Starting import of Co-Occurrence Matrix")
        cooc_matrix = pd.read_csv("Data/GloVe_CoOcMatrix.csv", index_col = 0)

    cooc_matrix = cooc_matrix + 1

    if set_seed is not None:
        np.random.seed(set_seed)

    prompt("Starting random initialization of parameters")
    target_matrix = np.random.normal(size = len(cooc_matrix.index) * word_embedding_dim, scale = 0.25).reshape(len(cooc_matrix.index), word_embedding_dim)
    context_matrix = np.random.normal(size = len(cooc_matrix.index) * word_embedding_dim, scale = 0.25).reshape(len(cooc_matrix.index), word_embedding_dim)
    bias_context_vec = np.random.normal(size = len(cooc_matrix.index), scale = 0.25).reshape(len(cooc_matrix.index), 1)
    bias_target_vec = np.random.normal(size=len(cooc_matrix.index), scale=0.25).reshape(len(cooc_matrix.index), 1)
    iota = np.ones((len(cooc_matrix.index), 1))

    target_matrix = np.array(target_matrix, dtype = np.float64)
    context_matrix = np.array(context_matrix, dtype=np.float64)

    def weighting(x, x_max, alpha):
        if x < x_max:
            return (x/x_max)**alpha
        else:
            return 1

    if use_weights:
        prompt("Starting the calculation of weights")
        weights = np.matrix([[weighting(float(e), x_max, alpha) for e in row] for row in cooc_matrix.values])
    else:
        weights = np.ones(cooc_matrix.shape)

    def calLoss(weights, cooc_matrix, target_matrix, context_matrix, bias_context_vec, bias_target_vec):
        """Calculates the loss"""
        J = np.matrix(np.multiply(weights, (np.matmul(target_matrix,  np.transpose(context_matrix)) + np.matmul(bias_context_vec, np.transpose(iota)) + np.matmul(bias_target_vec, np.transpose(iota)) - np.log(cooc_matrix.to_numpy()))**2)).flatten().sum()

        return J

    prompt("Starting calculation")
    loss_series = [calLoss(weights, cooc_matrix, target_matrix, context_matrix, bias_context_vec, bias_target_vec)]

    for e in tqdm(range(epochs)):
        C = -2 * np.multiply(weights, (np.log(cooc_matrix.to_numpy()) - np.matmul(target_matrix, np.transpose(context_matrix)) - np.matmul(bias_context_vec, np.transpose(iota)) - np.matmul(bias_target_vec, np.transpose(iota))))

        gradient_target = np.matmul(C, context_matrix)
        gradient_context = np.matmul(C, context_matrix)
        gradient_bias_context_vec = np.matmul(C, iota)
        gradient_bias_target_vec = np.matmul(C, iota)

        target_matrix -= eta * gradient_target
        context_matrix -= eta * gradient_context
        bias_target_vec -= eta_bias * gradient_bias_target_vec
        bias_context_vec -= eta_bias * gradient_bias_context_vec

        loss_series.append(calLoss(weights, cooc_matrix, target_matrix, context_matrix, bias_context_vec, bias_target_vec))

    plt.plot(pd.Series(loss_series).T)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Series")
    plt.show()

    return pd.DataFrame(context_matrix, index=cooc_matrix.index), pd.DataFrame(target_matrix, index =cooc_matrix.index)







