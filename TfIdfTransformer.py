import pandas
import pandas as pd
import numpy as np
from numpy.linalg import norm
import pickle
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk


class TfIdfTransformer:
    def __init__(self, df: pandas.DataFrame):
        nltk.download('punkt')

        self.df = df
        self.tfidf_matrix_array = np.load('tfidf.npy')

        with open("vectorizer.pkl", "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)

    def clean_text(self, text):
        """
        Function for clearing text from html-tags

        Parameters:
        ----------
        text : str
            Text for clearing

        Returns:
        -------
        clean_text : str
            Preprocessed text
        """
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # Convert to lowercase
        text = text.lower()

        # Tokenize the text
        tokens = word_tokenize(text)

        # Initialize Porter Stemmer
        porter = PorterStemmer()

        # Stemming and remove non-alphabetic characters
        clean_tokens = [porter.stem(token) for token in tokens if
                        token.isalpha()]

        # Join the tokens back into a single string
        clean_text = ' '.join(clean_tokens)

        return clean_text

    def cosine_similarity(self, matrix, vector):
        """
        Function for cosine similarity estimation

        Parameters:
        ----------
        matrix : numpy.ndarray
            tf-idf matrix for cosine similarity calculation

        vector : numpy.ndarray
            Transformed text to vector

        Returns:
        -------
        similarity : float
            Similarity value
        """
        # Compute the dot product between the matrix and the vector
        dot_product = np.dot(matrix, vector)

        # Compute the L2 norm of the matrix and the vector
        matrix_norm = np.linalg.norm(matrix, axis=1)
        vector_norm = np.linalg.norm(vector)

        # Compute cosine similarity
        similarity = dot_product / (matrix_norm * vector_norm)

        return similarity

    def get_recommends(self, wish: str):
        """
        Function for recommendation forming

        Parameters:
        ----------
        wish : str
            Description about hotel from user

        Returns:
        -------
        list[int]
            Resulting recommendations
        """
        tfidf_matrix_array_city_name = self.tfidf_matrix_array[
            list(self.df.index)]
        wish = self.clean_text(wish)
        wish_vector = self.tfidf_vectorizer.transform([wish]).toarray()[0]
        cosine_sim = self.cosine_similarity(
            tfidf_matrix_array_city_name, wish_vector
        )
        idx = np.argsort(cosine_sim)[::-1]
        return list(idx)
