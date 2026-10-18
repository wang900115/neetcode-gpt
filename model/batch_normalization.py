import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists

        eps = 1e-5

        batch_size  = len(x)
        num_features = len(x[0])

       
        if training:

            mean = [
                sum(x[i][j] for i in range(batch_size)) / batch_size
                for j in range(num_features)
            ]

            var = [
                sum((x[i][j] - mean[j]) ** 2 for i in range(batch_size))
                / batch_size
                for j in range(num_features)
            ]

            running_mean = [
                (1 - momentum) * running_mean[j] + momentum * mean[j]
                for j in range(num_features)
            ]

            running_var = [
                (1 - momentum) * running_var[j]
                + momentum * var[j]
                for j in range(num_features)
            ]
        else:
            mean = running_mean
            var = running_var
        y = [
            [
                gamma[j] * (x[i][j] - mean[j])
                / math.sqrt(var[j] + eps)
                + beta[j]
                for j in range(num_features)
            ]
            for i in range(batch_size)
        ]
        y = [[round(value, 4) for value in row] for row in y]
        running_mean = [round(value, 4) for value in running_mean]
        running_var = [round(value, 4) for value in running_var]
        return y, running_mean, running_var
