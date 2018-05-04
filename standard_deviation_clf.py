import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array


class StandardDeviationClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        sep_class = dict()
        mean_class = dict()
        std_class = dict()

        for c in self.classes_:
            sep_class[c] = [X[idx] for idx in filter(lambda i: y[i] == c, range(0, len(y)))]
            mean_class[c] = np.mean(sep_class[c])
            std_class[c] = np.std(sep_class[c])
        thresh_one = mean_class[-1] + std_class[-1]
        thresh_two = mean_class[1] - std_class[1]
        self.neg_max = min(thresh_one, thresh_two)
        self.pos_min = max(thresh_one, thresh_two)

        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        y_pred = []
        print(self.neg_max, self.pos_min)
        for x in X:
            if x <= self.neg_max:
                y_pred.append(-1)
            elif x <= self.pos_min:
                y_pred.append(0)
            else:
                y_pred.append(1)
        return y_pred