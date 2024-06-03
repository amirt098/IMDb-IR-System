import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

from .basic_classifier import BasicClassifier
from .data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

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
        self.X_train = x
        self.y_train = y
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
        y_pred = []

        for xi in tqdm(x, desc="Predicting"):
            distances = np.linalg.norm(self.X_train - xi, axis=1)
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_neighbors]
            y_pred.append(np.bincount(nearest_labels).argmax())

        return np.array(y_pred)

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
        y_pred = self.predict(x)
        report = classification_report(y, y_pred)
        return report


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    file_path = 'IMDB Dataset.csv'
    data_loader = ReviewLoader(file_path)
    X_train, y_train = data_loader.load_data()
    X_test, y_test = data_loader.load_data()

    knn_classifier = KnnClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    report = knn_classifier.prediction_report(X_test, y_test)
    print(report)
