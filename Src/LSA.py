from collections.abc import Iterable
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

def LSA(input_mat: Iterable):

    numb_topics = getNumbTopics(input_mat)

    lsa_model = TruncatedSVD(n_components=numb_topics, algorithm='arpack')
    doc_mat_lsa = lsa_model.fit_transform(input_mat)

    return doc_mat_lsa

def getNumbTopics(input_mat: Iterable):

    list_var_explained = []
    numb_topics = 0
    flag = True
    steps = [50, 100, 250, 500, 800, 1000, 1500, 2000, 2500]

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

    plt.plot(steps, list_variance_explained)
    plt.axhline(y=0.7, color='g', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel("Explained Variance")
    plt.title("Explained Variance by n-components")
    plt.show()

# sources used:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
# https://www.kaggle.com/rajmehra03/topic-modelling-using-lda-and-lsa-in-sklearn
# https://stackoverflow.com/questions/69091520/determine-the-correct-number-of-topics-using-latent-semantic-analysis
# https://medium.com/swlh/truncated-singular-value-decomposition-svd-using-amazon-food-reviews-891d97af5d8d
# (https://github.com/priyagunjate/Word-Vectors-using-Truncated-SVD/blob/master/Assignment11--version1.ipynb)