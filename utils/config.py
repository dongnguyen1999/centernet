import random
import os
import numpy as np
from pathlib import Path

class Config:
    def __init__(self, 
        num_classes, train_path, checkpoint_path, annotation_filename, data_base, 
        num_channels=3, name='keras_model', logging_base=None, valid_path=None, test_paths=[], image_id='image_id', weights_path=None,
        epochs=1, batch_size=1, lr=1e-4, seed = 2610, test_size=0.2, val_size=0.2, input_size=512) -> None:
        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.train_path = os.path.join(data_base, train_path) if train_path != None else train_path
        self.test_paths = [os.path.join(data_base, test_path) for test_path in test_paths]
        self.annotation_filename = annotation_filename
        self.valid_path = os.path.join(data_base, valid_path) if valid_path != None else valid_path
        self.lr = lr
        self.seed = seed
        self.test_size = test_size
        self.val_size = val_size
        self.data_base = data_base  
        self.input_size = input_size
        self.output_size = self.input_size // 4 # Center output size with stride 4 
        self.image_id = image_id
        self.weights_path = weights_path
        self.logging_base = data_base if logging_base == None else logging_base
        self.checkpoint_path = os.path.join(self.logging_base, checkpoint_path) if self.logging_base != None else checkpoint_path

        self.random_system()
    
    def random_system(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)



def get_project_root() -> Path:
    return Path(__file__).parent.parent