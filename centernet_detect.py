# from keras_centernet_count.models.decode import ClassificationDecode
# from keras_centernet_count.train import train
# from centernet_detect.eval import eval_models
from centernet_detect.train import train
from centernet_detect.models.hourglass import create_model
from centernet_detect.models.decode import _ctdet_decode, visualize
from centernet_detect.losses import compile_model
from centernet_detect.dataset.vn_vehicle import load_data, DataGenerator
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
    num_classes=3, 
    train_path='video1_all\\test', 
    valid_path='video1_all\\test',
    test_paths=['video1_all\\test'],
    checkpoint_path='models\hourglass_centernet_1stack_512\\rf_v1',
    annotation_filename='_annotations_custom_v2.txt',
    data_base='C:\\vehicle-data',
    epochs=1,
    batch_size=1,
    image_id='filename',
    input_size=512
    # weights_path='/kaggle/working/centernet.hdf5',
)
train_df, valid_df, test_df, le = load_data(config)

# print(estimate_crowd_threshold(train_df, le, config))

# data_gen = DataGenerator(valid_df, config, mode='valid')
# X, Y = data_gen.__getitem__(1)
# Y_pred = Y[:,:,:,: config.num_classes]

# hm, reg, wh = Y
# print(hm.shape, reg.shape, wh.shape)
# ds = _ctdet_decode(hm[0][..., :3].reshape((-1, 128, 128, 3)), reg[0][..., :2].reshape((-1, 128, 128, 2)), wh[0][..., :2].reshape((-1, 128, 128, 2)))
# visualize(ds[0], X[0], config, display=True)
# plt.show()

# out = _ctdet_decode(Y_pred)


# print(X.shape, Y_pred.shape)

# aug = data_augmentation()
# augX, augY = aug(images=[X[0].astype(np.float32)], heatmaps=[Y[0].astype(np.float32)])
# print(len(augX)
# )

# print(np.min(Y_pred[0][..., 0]))
# print(np.max(Y_pred[0][..., 0]))

# img = cv2.resize(X[0], (config.output_size,config.output_size))

# plt.imshow(img)
# plt.imshow(wh[0][..., 2], alpha=0.8)

# plt.show()


model = create_model(config, 1)
model.summary()
# model = CountDecode(model)
# y = model.predict(X)

# hm, reg, wh = y
# print(hm.shape, reg.shape, wh.shape)

# print(y.shape)

# out = Y_pred
# counts = []
# for i in range(3):  
#     counts.append(np.sum(out[:,:,:,i]))

# print(counts)
# print(true_counts)

# model = CountDecode(model, config.num_classes)
# y = model.predict(X)
# print(y)


train(model, train_df, valid_df, config, test_df=test_df, generator=DataGenerator)

# eval_models(valid_df, test_df, config, model_prefix='centernet_count_hg', 
#     model_garden={
#         '1stack': create_model(config, 1),
#         '2stack': create_model(config, 2),
#     }
# )


# maes = calcmAP(model, valid_df, config, confidences=[0.25, 0.5, 0.7])

# print(maes.shape)



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