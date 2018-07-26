import torch
from torch.nn import functional

class SoftmaxWithTemperature:
    def __init__(self, temperature):
        """
        formula: softmax(x/temperature)
        """
        self.temperature  = temperature

    def __call__(self, x, temperature=None):
        if not temperature is None:
            return functional.softmax(x / temperature, -1)
        else:
            print('temperature:{}'.format(self.temperature))
            return functional.softmax(x / self.temperature, -1)

CUDA_wrapper = lambda x: x
