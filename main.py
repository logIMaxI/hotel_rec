import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from numpy import dot
from numpy.linalg import norm
import pickle
import gzip
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

df = pd.read_csv('hotels_short.csv')
tfidf_matrix_array = np.load('tfidf.npy')

with open("vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

def get_tfidf(documents):
    """
    Function for tf-idf text coding

    Parameters:
    ----------
    documents : list[str]
        Corpus of text

    Returns:
    -------
    tfidf_vectorizer : TfidfVectorizer
        Model for creating tf-idf matrix from corpus of text

    tfidf_matrix_array : numpy.ndarray
        tf-idf matrix
    """
    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    # Fit and transform the documents
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    # Convert the TF-IDF matrix to an array
    tfidf_matrix_array = tfidf_matrix.toarray()
    # Get feature names (words)
    return tfidf_vectorizer, tfidf_matrix_array

def clean_text(text):
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
    clean_tokens = [porter.stem(token) for token in tokens if token.isalpha()]
    
    # Join the tokens back into a single string
    clean_text = ' '.join(clean_tokens)
    
    return clean_text

def cosine_similarity(matrix, vector):
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

def get_recommends(df, city: str, wish: str, tfidf_vectorizer, tfidf_matrix_array, k):
    """
    Function for recommendation forming

    Parameters:
    ----------
    df : pd.DataFrame
        Initial info about hotels

    city : str
        City filter

    wish : str
        Description about hotel from user

    tfidf_vectorizer : TfidfVectorizer
        Model for converting text into vector

    tfidf_matrix_array : numpy.ndarray
        tf-idf matrix for cosine similarity calculation

    k : int
        Count of recommendation

    Returns:
    -------
    list[int]
        Resultimg recommendations
    """
    df_city_name = df[df['cityName'] == city]
    tfidf_matrix_array_city_name = tfidf_matrix_array[list(df_city_name.index)]
    wish = clean_text(wish)
    wish_vector = tfidf_vectorizer.transform([wish]).toarray()[0]
    cosine_sim = cosine_similarity(tfidf_matrix_array_city_name, wish_vector)
    idx = np.argsort(cosine_sim)[::-1][:k]
    return list(df_city_name.iloc[idx].index)

def get_info_city(df):
    """
    Function for writing in console info about hotels

    Parameters:
    ----------
    df : pd.DataFrame
        Info about hotels
    """
    for i in range(len(df)):
      cityName = df.iloc[i]['cityName']
      countyName = df.iloc[i]['countyName']
      HotelName = df.iloc[i]['HotelName']
      PersCount = df.iloc[i]['PersCount']
      Address = df.iloc[i]['Address']
      Price = df.iloc[i]['Price']
      AreaSquare = df.iloc[i]['AreaSquare']

      print(f"Country Name : {countyName}")
      print(f"City Name : {cityName}")
      print(f"Hotel Name : {HotelName}")
      print(f"Mark : {PersCount}")
      print(f"Hotel Address : {Address}")
      print(f"Area Square : {AreaSquare}")
      print('-'*50)


def main():
    """
    Function for running model
    """
    k = 5

    print('City: ')
    city = str(input())

    print('Wish: ')
    wish = str(input())

    idx = get_recommends(df, city, wish, tfidf_vectorizer, tfidf_matrix_array, k)
    sorted_df = df.iloc[idx].sort_values(by=['PersCount'], ascending=False)
    get_info_city(sorted_df)


if __name__ == "__main__":
    main()
