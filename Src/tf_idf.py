from collections.abc import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def tf_idf(corpus: Iterable):

    tfidf_vec = TfidfVectorizer()
    tfidf = tfidf_vec.fit_transform(corpus)

    mat_tfidf = tfidf.toarray()
    df_tfidf = pd.DataFrame(data=mat_tfidf,columns=tfidf_vec.get_feature_names())

    return mat_tfidf, df_tfidf