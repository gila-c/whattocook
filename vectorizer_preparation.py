from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd

def vectorizer_preparation(bow_corpus):
    
    """
    Convert a collection of text documents to Document-Term and TF-IDF vectorizer representations  
    
    Arguments:
        bow_corpus: a list of documents, each entry is a string
    
    Return:
        d: a dictionary containing
            dtm: Document-Term matrix
            norm_dtm: normalized Document-Term matrix
            tfidf: TF-IDF matrix
            norm_tfidf: normalized TF-IDF matrix
            p: number of different words in total
    
    """
    
    # convert text to word count vectors with CountVectorizer.
    vec = CountVectorizer()
    X = vec.fit_transform(bow_corpus)
    dtm = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
    dtm = np.array(dtm)
    
    # normalized count vectorizer
    normalizer = Normalizer()
    norm_dtm = normalizer.fit_transform(dtm)
    
    # convert text to word frequency vectors with TfidfVectorizer.
    vectorizer = TfidfVectorizer() 
    tfidf = vectorizer.fit_transform(bow_corpus)
    tfidf = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
    
    # normalized TfidfVectorizer
    normalizer = Normalizer()
    norm_tfidf = normalizer.fit_transform(tfidf)
    
    # get the ingredients name
    ingredients=vec.get_feature_names()
    
    # number of ingredients
    p = len(vec.get_feature_names())
    
    d=dict()
    d["p"]=p
    d["ingredients"]=ingredients
    d["dtm"]=dtm
    d["norm_dtm"]=norm_dtm
    d["tfidf"]=tfidf
    d["norm_tfidf"]=norm_tfidf
    
    return(d)