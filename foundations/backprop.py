import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        pass

        #  forward
        res = np.dot(x, w) + b
        res_z = 1 / (1 + np.exp(-res))
        #  loss func
        loss = 0.5 * np.power((res_z - y_true),2)

        # cal err
        prediction_err = res_z - y_true

        # cal u
        omag = prediction_err * res_z * (1- res_z) * x

        # cal b
        omag_b = prediction_err  * res_z * (1 - res_z)

        return np.round(omag,5), np.round(omag_b,5)

