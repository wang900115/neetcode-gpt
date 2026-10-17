import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)

        std = math.sqrt(2 / (fan_in + fan_out))
        weights = torch.randn(fan_out,fan_in) * std

        return [
            [round(float(w),4) for w in row]
            for row in weights.tolist()
        ]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)

        std = math.sqrt(2 / (fan_in))
        weights = torch.randn(fan_out,fan_in) * std

        return [
            [round(float(w),4) for w in row]
            for row in weights.tolist()
        ]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)

        # First create all weight matrices so random numbers are drawn in sequence
        layers = []
        for layer_idx in range(num_layers):
            fan_in = input_dim if layer_idx == 0 else hidden_dim
            fan_out = hidden_dim

            if init_type == "xavier":
                std = math.sqrt(2.0 / (fan_in + fan_out))
            elif init_type == "kaiming":
                std = math.sqrt(2.0 / fan_in)
            elif init_type == "random":
                std = 1.0
            else:
                raise ValueError("init_type must be 'xavier', 'kaiming', or 'random'")

            weights = torch.randn(fan_out, fan_in) * std
            layers.append(weights)

        # Create input vector of shape (input_dim,) after weight generation
        x = torch.randn(input_dim)

        activations_std = []
        for weights in layers:
            x = F.relu(x @ weights.T)
            activations_std.append(round(x.std().item(), 2))

        return activations_std
