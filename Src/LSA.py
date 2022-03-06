from collections.abc import Iterable
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

def LSA(input_mat: Iterable):
    """
    Function which generates a LSA representation of the corpus.

    :param corpus: Iterable/list/array or pd.Series which holds the tf-idf matrix
    :return: LSA matrix
    """

    numb_topics = getNumbTopics(input_mat)

    lsa_model = TruncatedSVD(n_components=numb_topics, algorithm='arpack')
    doc_mat_lsa = lsa_model.fit_transform(input_mat)

    return doc_mat_lsa

def getNumbTopics(input_mat: Iterable):
    """
    Function which determines the appropriate number of components.

    :param input_mat: Iterable/list/array or pd.Series which holds the tf-idf matrix
    :return: Number of components
    """

    list_var_explained = []
    numb_topics = 0
    flag = True
    steps = [50, 100, 250, 500, 800, 1000, 1200]

    for n in steps:

        lsa_model = TruncatedSVD(n_components=n, algorithm='arpack')
        doc_mat_lsa = lsa_model.fit(input_mat)
        list_var_explained.append(doc_mat_lsa.explained_variance_ratio_.sum())

        if(doc_mat_lsa.explained_variance_ratio_.sum() >= 0.7 and flag):
            flag = False
            numb_topics = n

    plotExplainedVariance(steps, list_var_explained)

    return numb_topics

def plotExplainedVariance(steps, list_variance_explained):
    """
    Function which plots the explained variance by n-components.

    :param steps: List which holds different amounts of topics
    :param list_variance_explained: List which holds the different explained variances
    """

    plt.plot(steps, list_variance_explained)
    plt.axhline(y=0.7, color='g', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel("Explained Variance")
    plt.title("Explained Variance by n-components")
    plt.show()