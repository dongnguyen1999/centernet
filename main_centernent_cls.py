from keras_centernet_cls.dataset.coco import downloadDataset
from keras_centernet_cls.eval import eval_models
from keras_centernet_cls.models.cnn import create_model
from keras_centernet_cls.metrics import calcScore
from keras_centernet_cls.train import train
from keras_centernet.models.decode import _ctdet_decode, visualize
from keras_centernet.metrics import calcmAP
from keras_centernet.losses import compile_model
from keras_centernet_cls.dataset.vn_vehicle import estimate_crowd_threshold, load_data, DataGenerator
from utils.config import Config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import sklearn.metrics
import keras
from keras.callbacks import Callback, ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")
import random

config = Config(
    name='hourglass_centernet_vehicle_v1',
    num_classes=2, 
    train_path='video1_all\\test', 
    valid_path='video1_all\\test',
    test_paths=['video1_all\\test'],
    checkpoint_path='models\hourglass_centernet_1stack_512\\rf_v1',
    annotation_filename='_annotations_custom_v2.txt',
    data_base='C:\\vehicle-data',
    epochs=50,
    batch_size=1,
    image_id='filename',
)

# train_df, valid_df, test_df, le = load_data(config)

# print(estimate_crowd_threshold(train_df, le, config))

# data_gen = DataGenerator(valid_df, le, 30, config, mode='valid')
# for X, y in data_gen:
#     if y[0,0] == 1:
#         img = X
#         # plt.imshow(X[0])
#         # plt.show()
#         break

# model = create_model(config, architecture='pretrained_resnet50', freeze_feature_block=False)
# model.summary()

# train(model, train_df, valid_df, le, 30, config, test_df=test_df, generator=DataGenerator)
# eval(model, train_df, valid_df, le, 30, config, test_df=test_df, generator=DataGenerator)
# eval_models(valid_df, test_df, le, 20, config, model_prefix='crowd_cls', 
#     model_garden={
#         'resnet50_fineturning': create_model(config, architecture='pretrained_resnet50', freeze_feature_block=False),
#         'vgg19': create_model(config, architecture='pretrained_vgg19'),
#         'vgg16': create_model(config, architecture='pretrained_vgg16'),
#         'resnet152': create_model(config, architecture='pretrained_resnet152'),
#         'resnet101': create_model(config, architecture='pretrained_resnet101'),
#         'resnet50': create_model(config, architecture='pretrained_resnet50'),
#         'mobilenetv2': create_model(config, architecture='pretrained_mobilenetv2'),
#         'inceptionv3': create_model(config, architecture='pretrained_inceptionv3'),
#     }
# )

downloadDataset(config)