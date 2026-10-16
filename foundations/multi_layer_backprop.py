import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        z1 = [
        sum(w * xi for w, xi in zip(weights, x)) + bias
        for weights, bias in zip(W1, b1)
        ]

        # ReLU activation
        a1 = [max(0.0, value) for value in z1]

        # Forward propagation: hidden layer -> output layer
        y_pred = [
            sum(w * ai for w, ai in zip(weights, a1)) + bias
            for weights, bias in zip(W2, b2)
        ]

        # MSE loss
        num = len(y_pred)
        loss = sum(
            (pred - true) ** 2
            for pred, true in zip(y_pred, y_true)
        ) / num

        # Gradient of loss w.r.t. predictions
        delta2 = [
            2 * (pred - true) / num
            for pred, true in zip(y_pred, y_true)
        ]

        # Gradients of W2 and b2
        dW2 = [
            [delta * ai for ai in a1]
            for delta in delta2
        ]
        db2 = delta2[:]

        # Backpropagation through W2 and ReLU
        delta1 = [
            sum(W2[j][i] * delta2[j] for j in range(len(W2)))
            * (1.0 if z1[i] > 0 else 0.0)
            for i in range(len(z1))
        ]

        # Gradients of W1 and b1
        dW1 = [
            [delta * xi for xi in x]
            for delta in delta1
        ]
        db1 = delta1[:]

        # Round every numeric value to 4 decimal places
        return {
            "loss": round(loss, 4),
            "dW1": [
                [round(value, 4) for value in row]
                for row in dW1
            ],
            "db1": [round(value, 4) for value in db1],
            "dW2": [
                [round(value, 4) for value in row]
                for row in dW2
            ],
            "db2": [round(value, 4) for value in db2],
        }
