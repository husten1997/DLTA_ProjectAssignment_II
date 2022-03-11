from collections.abc import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from Src.prompt import prompt

def tf_idf(corpus: Iterable):
    """
    Function which generates a tf-idf representation of the corpus.

    :param corpus: Iterable/list/array or pd.Series which holds the document-texts
    :return: tf-idf matrix as dataframe
    """

    prompt("Starting TfIdf algorithm")
    tfidf_vec = TfidfVectorizer()
    tfidf = tfidf_vec.fit_transform(corpus)

    mat_tfidf = tfidf.toarray()
    df_tfidf = pd.DataFrame(data=mat_tfidf, columns=tfidf_vec.get_feature_names(), index=corpus.index)

    return mat_tfidf, df_tfidf