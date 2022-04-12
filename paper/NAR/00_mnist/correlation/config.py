import pandas as pd


args = {
    
    "color_list": ['#1300ff','#ff0c00','#25ff00', '#d000ff','#e2ff00', '#00fff6', '#ff8800', '#fccde5','#178b66', '#8a0075'],
    "mnist_labels_dict" : {0: '0', 1: '1',2: '2', 3:'3',4: '4', 5: '5', 6: '6', 7:'7', 8:'8', 9:'9'},
    "fmnist_labels_dict" : {0: 'T-shirt/top',1: 'Trouser',2: 'Pullover', 3: 'Dress',4: 
                            'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'},
    
    "mnist_data_save_folder" : '/home/shenwanxiang/data/mnist/correlation/',
    "fmnist_data_save_folder" : '/home/shenwanxiang/data/fmnist/correlation/',    
    
    "results_save_folder": './results_data',

    "metric":"correlation",
    "var_thr":0,
    'seed': 888,
    'epochs': 100,
    'batch_size': 64,
    'conv1_kernel_size':3,
    'dense_layers':[128, 64],
    'lr': 0.0001,
    'dense_avf': 'relu',
    'last_avf': 'softmax',
    'patience': 1000}


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
    
    
P = Struct(**args)


