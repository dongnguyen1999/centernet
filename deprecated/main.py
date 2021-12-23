from crowd_classification.train import train
from crowd_classification.models.vgg import create_model
from crowd_classification.dataset.vn_vehicle import load_data, preprocessing
from utils.loss_functions import CenterNetLosses
# from models.hourglass_centernet import create_model
from utils.metrics import calculate_map
from numpy.lib.shape_base import expand_dims
from utils.config import Config
# from utils.dataset.wheat import load_data, test_dataset
# from utils.dataset.vn_vehicle import DataGenerator, load_data, test_dataset, preprocessing
# from utils.dataset.detrac import DataGenerator, load_data, test_dataset, preprocessing
import matplotlib.pyplot as plt
# from utils.train import train
from utils.output_decoder import OutputDecoder
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import os

config = Config(
    name='crowd_classification',
    num_classes=2, 
    train_path='crowd_train\\train',
    test_path='crowd_train\\test',
    valid_path='crowd_train\\valid',
    checkpoint_path='models\\crowd_classification\\v1',
    annotation_filename='',
    data_base='C:\\vehicle-data',
    epochs=300,
    batch_size=8,
    image_id='filename',
    lr=0.001,
    momentum=0.9,
    input_size=200,
    # weights_path='best_map\\02-0.815.hdf5',
)


# preprocessing(config)
train_gen, valid_gen, test_gen = load_data(config)

model = create_model(config)

# # fit model
train(model, train_gen, valid_gen, test_gen, config)