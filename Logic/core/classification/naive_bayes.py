import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc

        Returns
        -------
        self
            Returns self as a classifier
        """
        x_counts = self.cv.fit_transform(x)

        self.number_of_samples, self.number_of_features = x_counts.shape
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.num_classes = len(self.classes)

        self.prior = class_counts / self.number_of_samples

        # Calculate feature counts and class conditional probabilities
        self.feature_counts = np.zeros((self.num_classes, self.number_of_features))
        for i, label in enumerate(self.classes):
            class_indices = np.where(y == label)
            self.feature_counts[i, :] = np.sum(x_counts[class_indices], axis=0)

        self.feature_probabilities = (self.feature_counts + self.alpha) / (
                np.sum(self.feature_counts, axis=1, keepdims=True) + self.alpha * self.number_of_features
        )

        # Calculate log probabilities for faster computation
        self.log_probs = np.log(self.feature_probabilities)

        return self

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        x_counts = self.cv.transform(x)
        log_likelihoods = np.dot(x_counts, self.log_probs.T) + np.log(self.prior)
        return self.classes[np.argmax(log_likelihoods, axis=1)]

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        pass

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        pass


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    path = 'IMDB Dataset.csv'
    loader = ReviewLoader(path)
    reviews, labels = loader.load_data()

    train_reviews, test_reviews, train_labels, test_labels = train_test_split(
        reviews, labels, test_size=0.2, random_state=42
    )

    nb_classifier = NaiveBayes(2)
    nb_classifier.fit(train_reviews, train_labels)

    print("Classification Report:")
    print(nb_classifier.prediction_report(test_reviews, test_labels))