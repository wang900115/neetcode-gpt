import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        vocab = sorted(set(
            word
            for sentence in positive+negative
            for word in sentence.split()
        ))

        word_to_id = {
            word: idx +1 
            for idx, word in enumerate(vocab)
        }
        # 2. Encode each sentence by replacing words with their IDs
        tensors = []
        for sentence in positive + negative:
            ids = [word_to_id[word] for word in sentence.split()]
            tensors.append(torch.tensor(ids,dtype=torch.long))
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        result = nn.utils.rnn.pad_sequence(
            tensors,
            batch_first = True,
            padding_value = 0
        )
        
        return result
