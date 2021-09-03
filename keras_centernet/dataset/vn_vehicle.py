from keras_centernet.utils import heatmap, normalize_image
from utils.config import Config
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.utils import Sequence
import cv2


def load_data(config: Config):
    names=['filename', 'x1', 'y1', 'x2', 'y2', 'label']
    train_df = pd.read_csv(os.path.join(config.train_path, config.annotation_filename), names=names)    
    valid_df = pd.read_csv(os.path.join(config.valid_path, config.annotation_filename), names=names)
    names.append('test_id')
    test_dfs = []
    for test_id, test_path in enumerate(config.test_paths):
        temp_df = pd.read_csv(os.path.join(test_path, config.annotation_filename), names=names)
        temp_df['test_id'] = test_id+1
        test_dfs.append(temp_df)

    test_df = test_dfs[0] if len(test_dfs) == 1 else pd.concat(test_dfs)

    le = LabelEncoder()
    train_df = train_df[train_df.label != 'person']
    train_df.label = le.fit_transform(train_df.label)
    test_df.label = le.transform(test_df.label)
    valid_df.label = le.transform(valid_df.label)

    print('Train data size: %d' % len(train_df.filename.unique()))
    print('Valid data size: %d' % len(valid_df.filename.unique()))
    print('Test data size: %d' % len(test_df.filename.unique()))

    return train_df, valid_df, test_df, le

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

    def __init__(self, dataframe, config: Config, mode='fit', shuffle=True): # mode: fit, predict, valid, eval
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
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        
        X = self.__generate_X(list_IDs_batch)
        
        if self.mode == 'predict':
            return X
        else:
            y = self.__generate_y(list_IDs_batch)
            return X, y
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)
    
    def __generate_X(self, list_IDs_batch):
        'Generates data containing batch_size samples'
        X = []
        
        for i, ID in enumerate(list_IDs_batch):
            im_name = os.path.basename(ID)
            img_path = os.path.join(self.base_path, im_name)
            img = self.__load_rgb(img_path)
            im_h, im_w = img.shape[:2]
            self.image_height, self.image_width = im_h, im_w
            img = cv2.resize(img, (self.input_size, self.input_size))
            X.append(img)

        X = np.array(X)
        return X
    
    def __generate_y(self, list_IDs_batch):
        output_layers = []
        for i, ID in enumerate(list_IDs_batch):
            # print(self.df)
            bbox = self.df[self.df[self.image_id]==ID][['x1', 'y1', 'x2', 'y2', 'label']].values
            output_layer = heatmap(bbox, (self.image_height, self.image_width), self.config)
            output_layers.append(output_layer)
        
        output_layers = np.array(output_layers)
        return output_layers
    
    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img
    
    def __load_rgb(self, img_path):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = normalize_image(img)
        return img