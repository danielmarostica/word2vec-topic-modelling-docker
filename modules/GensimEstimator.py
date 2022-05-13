from sklearn.base import TransformerMixin, BaseEstimator
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

class SentenceVectorizer(BaseEstimator, TransformerMixin):

    '''
    Transforms data into a DataFrame of Word2Vec sentence vectors.
    Is basically a Gensim Word2Vec class adapted for sentences.
    '''

    def __init__(self, min_count=5, window=7, vector_size=200, workers=16, sg=False):
        self.min_count = min_count
        self.window = window
        self.vector_size = vector_size
        self.workers = workers
        self.sg = sg

    def sent_vectorizer(self, sentence, vectorizer):
        sent_vec =[]
        numw = 0
        for word in sentence:
            try:
                if numw == 0:
                    sent_vec = vectorizer.wv[word]  
                else:
                    sent_vec = np.add(sent_vec, vectorizer.wv[word])
                numw += 1

            except Exception as e: # if word not present
                if numw == 0:
                    sent_vec = np.zeros(self.vector_size)
                else:
                    sent_vec = np.add(sent_vec, np.zeros(self.vector_size))

        if numw > 0:
            return np.asarray(sent_vec) / numw
        else:
            return np.zeros(self.vector_size)

    def splitter(self, X):
        if isinstance(X, pd.DataFrame):
            return X.text.apply(str.split)
        else:
            return X.apply(str.split)

    def fit(self, X):
        split_X = self.splitter(X)
        self.w2v = Word2Vec(split_X, min_count=self.min_count, window=self.window, vector_size=self.vector_size, workers=self.workers)
        return self

    def transform(self, X):
        split_X = self.splitter(X)
        X_vec=[]
        for sentence in split_X:
            X_vec.append(self.sent_vectorizer(sentence, self.w2v))

        return pd.DataFrame(X_vec)


