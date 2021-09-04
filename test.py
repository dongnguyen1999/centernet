from keras_centernet.models.decode import _ctdet_decode, visualize
from keras_centernet.metrics import calcmAP
from keras_centernet.losses import compile_model
from keras_centernet.dataset.vn_vehicle import load_data, DataGenerator
from utils.config import Config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import sklearn.metrics
from keras_centernet.models.hourglass import HourglassNetwork
import keras
from keras.callbacks import Callback, ModelCheckpoint
from utils.train import train

import warnings
warnings.filterwarnings("ignore")
import random

config = Config(
    name='hourglass_centernet_vehicle_v1',
    num_classes=3, 
    train_path='video1_all\\test', 
    valid_path='video1_all\\test',
    test_paths=['test\\test1', 'test\\test2'],
    checkpoint_path='models\hourglass_centernet_1stack_512\\rf_v1',
    annotation_filename='_annotations_custom_v2.txt',
    data_base='C:\\vehicle-data',
    epochs=50,
    batch_size=1,
    image_id='filename',
    # weights_path='/kaggle/working/centernet.hdf5',
)

train_df, valid_df, test_df, le = load_data(config)

# data_gen = DataGenerator(valid_df, config, mode='valid')
# X, Y = data_gen.__getitem__(1)
# print(Y.shape)

# img = cv2.resize(X[0], (config.output_size,config.output_size))

# plt.imshow(img)
# plt.imshow(Y[0][..., 5], alpha=0.5)

# plt.show()

kwargs = {
  'num_stacks': 1,
  'cnv_dim': 256,
  'weights': 'ctdet_coco',
  'inres': (config.input_size, config.input_size),
}
heads = {
  'hm': 3,
  'reg': 2,
  'wh': 2
}

model = HourglassNetwork(heads=heads, **kwargs)
model.summary()


train(model, train_df, valid_df, config, test_df=test_df, generator=DataGenerator)

# calcmAP(model, valid_df, config)

# Y = Y[..., :config.num_classes+4]
# print(Y.shape)

# hm = Y[..., : config.num_classes]
# reg = Y[..., config.num_classes: config.num_classes+2]
# wh = Y[..., config.num_classes+2: config.num_classes+4]

# detections = _ctdet_decode(hm, reg, wh)
# print(detections)

# visualize(detections[0], X[0], config, display=True)
# plt.show()