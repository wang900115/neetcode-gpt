import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        pred = np.clip(y_pred,1e-7, 1 - 1e-7)
        return np.round(-np.mean(y_true * np.log(pred) + (1-y_true) * np.log(1-pred)), 4 )


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        pred = np.clip(y_pred,1e-7, 1 - 1e-7)

        total_loss = np.sum(y_true * np.log(pred))

        loss = -total_loss / y_true.shape[0]

        return np.round(loss,4)
