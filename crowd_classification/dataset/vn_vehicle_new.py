from utils.augmentor.misc import MiscEffect
from utils.augmentor.color import VisualEffect
from centernet_detect.utils import heatmap, normalize_image
from utils.config import Config
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.utils import Sequence
import cv2
from imgaug import augmenters
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from glob import glob

def load_data(config: Config):
    train_gen = DataGenerator(config.train_path, config, mode='fit')
    valid_gen = DataGenerator(config.valid_path, config, mode='valid')
    test_gen = DataGenerator(config.test_path, config, mode='eval')
    return train_gen, valid_gen, test_gen

def load_directory(path):
    if not os.path.exists(path):
        raise ValueError("Input path is not exist!")

    df = []
    for subdir in glob(os.path.join(path, '*/')):
        label_name = os.path.basename(subdir[:-1])
        for img_file in glob(os.path.join(subdir, '*.jpg')):
            df.append([img_file, label_name])
    
    return pd.DataFrame(df, columns=['filename', 'label'])
        


class DataGenerator(Sequence):
    'Generates data for Keras'
    # def __init__(self, list_IDs, df, config: Config, target_df=None, mode='fit',
    #              base_path=config.IMAGE_PATH, image_paths=None,
    #              batch_size=4, dim=(128, 128), n_channels=3,
    #              n_classes=3, random_state=config.seed, shuffle=True):
    #     self.dim = dim
    #     self.batch_size = batch_size
    #     self.df = df
    #     self.mode = mode
    #     self.base_path = base_path
    #     self.target_df = target_df
    #     self.list_IDs = list_IDs
    #     self.n_channels = n_channels
    #     self.n_classes = n_classes
    #     self.shuffle = shuffle
    #     self.random_state = random_state
    #     self.image_paths = image_paths
        
    #     self.on_epoch_end()

    def __init__(self, path, config: Config, mode='fit', shuffle=True): # mode: fit, predict, valid, eval
        print(f'Loaded directory {path} for {mode} dataset')
        dataframe = load_directory(path)
        print(f'Loaded {len(dataframe)} images')
        print(f'Label 0: {len(dataframe[dataframe.label == "0"])} images')
        print(f'Label 1: {len(dataframe[dataframe.label == "1"])} images')
        self.config = config
        self.image_ids = dataframe[config.image_id].unique()
        self.df = dataframe
        self.train_path = config.train_path 
        self.valid_path = config.valid_path
        self.test_paths = config.test_paths
        if self.valid_path == None:
            self.valid_path = config.train_path
        self.batch_size = config.batch_size
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.is_train = mode == 'fit'
        self.num_classes = config.num_classes
        self.seed = config.seed
        self.enable_augmentation = config.enable_augmentation
        self.visual_effect = VisualEffect(
            color_factor=0.5,
            contrast_factor=0.5,
            brightness_factor=0.5,
            sharpness_factor=0.5)
        self.misc_effect = MiscEffect()

        self.dim = (config.input_size, config.input_size)
        df_modes = {
            'fit': self.train_path,
            'valid': self.valid_path,
            'eval': self.test_paths,
        }

        self.mode = mode
        self.base_path = df_modes[mode] 
        self.list_IDs = dataframe[config.image_id].unique()
        self.n_channels = config.num_channels
        self.n_classes = config.num_classes
        self.shuffle = shuffle
        self.random_state = config.seed
        self.image_id = config.image_id

        self.num_output_layers = self.num_classes + 4 # The first n layers is heatmap for each class, 2 next layers for h-offset and w-offset, 2 last layers for h-size and w-size
        self.steps_factor = config.steps_factor if self.mode == 'fit' and config.steps_factor != None else 1

        np.random.seed(self.random_state)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) * self.steps_factor/ self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        if self.steps_factor < 0.5:
            self.on_epoch_end()

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        
        X = self.__generate_X(list_IDs_batch)
        
        if self.mode == 'predict':
            return X
        else:
            X, y = self.__generate_xy(list_IDs_batch)
            return X, y
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __generate_X(self, list_IDs_batch):
        'Generates data containing batch_size samples'
        X = []
        
        for i, ID in enumerate(list_IDs_batch):
            # im_name = os.path.basename(ID)
            # img_path = os.path.join(self.base_path, im_name)
            img = cv2.imread(ID)
            img = normalize_image(img)
            img = cv2.resize(img, (self.input_size, self.input_size))
            X.append(img)

        X = np.array(X)
        return X
    
    def __generate_xy(self, list_IDs_batch):
        X = []
        Y = []
        for i, ID in enumerate(list_IDs_batch):
            # im_name = os.path.basename(ID)
            # img_path = os.path.join(self.base_path, im_name)

            img = cv2.imread(ID)
            label = self.df[self.df[self.image_id]==ID].label

            if self.enable_augmentation:
                img = self.visual_effect(img)
                img, _ = self.misc_effect(img, np.array([[1,1,1,1]], dtype=np.int32))

            img = normalize_image(img)
            img = cv2.resize(img, (self.input_size, self.input_size))
            X.append(img)
            Y.append(int(label))
        
        X = np.array(X)
        Y = np.array(Y)

        return X, Y