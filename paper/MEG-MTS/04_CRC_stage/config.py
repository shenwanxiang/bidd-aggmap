import pandas as pd


args = {
    'epochs': 50,
    'batch_size': 2,
    'conv1_kernel_size':13}


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
    
    
P = Struct(**args)