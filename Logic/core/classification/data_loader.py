import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..word_embedding import FastText


class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = FastText()
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        data = pd.read_csv(self.file_path)

        self.review_tokens = [self.fasttext_model.model.tokenize(review) for review in data['review']]
        self.sentiments = data['sentiment'].values

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        self.embeddings = [self.fasttext_model.get_query_embedding(tokens) for tokens in
                           tqdm.tqdm(self.review_tokens)]

    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(self.embeddings), np.array(self.sentiments),
            test_size=test_data_ratio, random_state=42
        )
        return x_train, x_test, y_train, y_test
