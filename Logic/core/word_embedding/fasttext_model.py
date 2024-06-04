import sys

import fasttext
import re
import string
sys.path.append("../")
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance

from fasttext_data_loader import FastTextDataLoader


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                    punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    if lower_case:
        text = text.lower()

    if punctuation_removal:
        text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)

    if stopword_removal:
        stop_words = set(stopwords.words('english')).union(set(stopwords_domain))
        tokens = [word for word in tokens if word not in stop_words]

    tokens = [word for word in tokens if len(word) >= minimum_length]

    return ' '.join(tokens)


class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None

    def train(self, texts):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        preprocessed_texts = [preprocess_text(text) for text in texts]

        with open('fasttext_training_data.txt', 'w') as f:
            for text in preprocessed_texts:
                f.write(text + '\n')

        self.model = fasttext.train_unsupervised('fasttext_training_data.txt', model=self.method)

    def get_query_embedding(self, query):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded. Please train or load a model before using this method.")
        preprocessed_query = preprocess_text(query)
        return self.model.get_sentence_vector(preprocessed_query)

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded. Please train or load a model before using this method.")

        # Obtain word embeddings for the words in the analogy
        vec1 = self.model.get_word_vector(word1)
        vec2 = self.model.get_word_vector(word2)
        vec3 = self.model.get_word_vector(word3)

        # Perform vector arithmetic
        analogy_vector = vec1 - vec2 + vec3

        # Create a dictionary mapping each word in the vocabulary to its corresponding vector
        words = self.model.get_words()
        word_vectors = {word: self.model.get_word_vector(word) for word in words}

        # Exclude the input words from the possible results
        exclude_words = {word1, word2, word3}

        # Find the word whose vector is closest to the result vector
        closest_word = None
        min_distance = float('inf')

        for word, vector in word_vectors.items():
            if word in exclude_words:
                continue

            dist = distance.cosine(analogy_vector, vector)

            if dist < min_distance:
                min_distance = dist
                closest_word = word

        return closest_word

    def save_model(self, path='FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded. Please train or load a model before saving.")
        self.model.save_model(path)

    def load_model(self, path="FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)

    def prepare(self, dataset, mode, save=False, path='FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_model(path)
        if save:
            self.save_model(path)


if __name__ == "__main__":
    ft_model = FastText(method='skipgram')

    path = '../classification/IMDB Dataset.csv'
    ft_data_loader = FastTextDataLoader(path)

    X = ft_data_loader.create_train_data()

    ft_model.train(X)
    ft_model.prepare(None, mode="save")

    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'queen'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "man"
    word2 = "king"
    word3 = "queen"
    print(
        f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
