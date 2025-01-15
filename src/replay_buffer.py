from random import sample
from numpy import array as nparray
from torch import Tensor as torchTensor

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity 
        self.data = [None] * capacity
        self.nb_stored = 0
        self.index = 0
    
    def append(self, s, a, r, s_, d):
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
        self.nb_stored = min(self.nb_stored + 1, self.capacity)

    def sample(self, batch_size):
        batch = sample(self.data[:self.nb_stored], batch_size)
        return list(map(lambda x:torchTensor(nparray(x)), list(zip(*batch))))
    
    def __len__(self): return self.nb_stored