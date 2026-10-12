import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        vocabulary = set()
        for sentence in positive:
            for word in sentence.split():
                vocabulary.add(word)
        
        for sentence in negative:
            for word in sentence.split():
                vocabulary.add(word)
        
        
        sorted_list = sorted(list(vocabulary))
        word_to_int = {}
        for i in range(len(sorted_list)):
            word_to_int[sorted_list[i]] = i + 1
        
        tensors = []
        for sentence in positive:
            curr_list = []
            for word in sentence.split():
                curr_list.append(word_to_int[word])
            tensors.append(torch.tensor(curr_list))
        
        for sentence in negative:
            curr_list = []
            for word in sentence.split():
                curr_list.append(word_to_int[word])
            tensors.append(torch.tensor(curr_list))

        return nn.utils.rnn.pad_sequence(tensors,batch_first=True)
            
