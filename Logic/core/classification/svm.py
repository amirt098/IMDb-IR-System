import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from .basic_classifier import BasicClassifier
from .data_loader import ReviewLoader


class SVMClassifier(BasicClassifier):
    def __init__(self):
        super().__init__()
        self.model = SVC()

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(x, y)

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
        return self.model.predict(x)

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
        return classification_report(y, y_pred)


# F1 accuracy : 78%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader(file_path='path_to_your_file.csv')
    loader.load_data()
    loader.get_embeddings()

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = loader.split_data(test_data_ratio=0.2)

    # Initialize and train SVM classifier
    svm_classifier = SVMClassifier()
    svm_classifier.fit(x_train, y_train)

    # Predict and evaluate
    print("Classification Report:")
    print(svm_classifier.prediction_report(x_test, y_test))