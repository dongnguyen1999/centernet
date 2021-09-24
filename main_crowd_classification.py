from crowd_classification.train import train
from crowd_classification.models.cnn import create_model
from crowd_classification.dataset.vn_vehicle import load_data
# from models.hourglass_centernet import create_model
from numpy.lib.shape_base import expand_dims
from utils.config import Config
# from utils.dataset.wheat import load_data, test_dataset
# from utils.dataset.vn_vehicle import DataGenerator, load_data, test_dataset, preprocessing
# from utils.dataset.detrac import DataGenerator, load_data, test_dataset, preprocessing
import matplotlib.pyplot as plt
# from utils.train import train
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import os

config = Config(
    name='crowd_classification',
    num_classes=2, 
    train_path='crowd_train\\test',
    test_path='crowd_train\\test',
    valid_path='crowd_train\\valid',
    checkpoint_path='models\\crowd_classification_512_s0\\v1',
    annotation_filename='',
    data_base='C:\\vehicle-data',
    epochs=1,
    batch_size=1,
    lr=0.0001,
    input_size=512,
    enable_augmentation=True,
    image_id='filename',
    # weights_path='best_map\\02-0.815.hdf5',
)


# preprocessing(config)
train_gen, valid_gen, test_gen = load_data(config)

# for i in range(1, 10):
#     X, Y = train_gen.__getitem__(i)
#     # hm, reg, wh = Y
#     print(X.shape, Y.shape)
#     print(Y)
#     # img = cv2.resize(X[0], (config.output_size,config.output_size))
#     plt.imshow(X[0])
#     # plt.imshow(hm[0][..., 0], alpha=0.5)
#     plt.show()

model = create_model(config, architecture='pretrained_vgg16', freeze_feature_block=True)

# # # fit model
train(model, train_gen, valid_gen, test_gen, config)