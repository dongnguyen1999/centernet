from utils.loss_functions import CenterNetLosses
from models.hourglass_centernet import create_model
from utils.metrics import calculate_map
from numpy.lib.shape_base import expand_dims
from utils.config import Config
# from utils.dataset.wheat import load_data, test_dataset
# from utils.dataset.vn_vehicle import DataGenerator, load_data, test_dataset
from utils.dataset.detrac import DataGenerator, load_data, test_dataset, preprocessing
import matplotlib.pyplot as plt
from utils.train import train
from utils.output_decoder import OutputDecoder

# config = Config(
#     name='centernet_wheat_v1',
#     num_classes=1, 
#     train_path='train', 
#     test_path='test',
#     checkpoint_path='models\centernet\\v1',
#     annotation_filename='train.csv',
#     data_base='D:\CenterNet\data\wheat',
#     epochs=5
# )

# config = Config(
#     name='hourglass_centernet_vehicle_v1',
#     num_classes=3, 
#     train_path='full_train\\train', 
#     valid_path='video1_all\\valid',
#     test_path='video1_all\\test',
#     checkpoint_path='models\hourglass_centernet_1stack_512\\rf_v1',
#     annotation_filename='_annotations_custom_v2.txt',
#     data_base='C:\\vehicle-data',
#     epochs=50,
#     batch_size=1,
#     image_id='filename',
#     # weights_path='/kaggle/working/centernet.hdf5',
# )

# config = Config(
#     name='hourglass_centernet_vehicle_v1',
#     num_classes=3, 
#     train_path='video1_all\\train', 
#     valid_path='video1_all\\valid',
#     test_path='video1_all\\test',
#     checkpoint_path='models\hourglass_centernet_1stack_512\\rf_v1',
#     annotation_filename='_annotations_custom_v2.txt',
#     data_base='C:\\vehicle-data',
#     epochs=50,
#     batch_size=1,
#     image_id='filename',
#     weights_path='best_map\\02-0.815.hdf5',
# )

config = Config(
    name='hourglass_centernet_vehicle_v1',
    num_classes=13, 
    train_path='detrac',
    test_path=None,
    checkpoint_path='models\hourglass_centernet_1stack_512_detrac\\v1',
    annotation_filename='_annotations.csv',
    data_base='C:\\vehicle-data',
    logging_base='D:\\',
    epochs=50,
    batch_size=1,
    image_id='filename',
    # weights_path='best_map\\02-0.815.hdf5',
)

# preprocessing(config)

# train_df, test_df = load_data(config)
train_df, test_df, valid_df, le = load_data(config)

# x1, y1 = test_dataset(train_df, config)
# x2, y2 = test_dataset(test_df, config, mode='test')
# plt.show()


# centernet_loss, heatmap_loss, offset_loss, size_loss = centernet_losses(config)
# loss = size_loss(y1, y2)
# print(loss)


# decoder = OutputDecoder(config)
# score_boxes = decoder.decode_y_true(y1)
# score_boxes = decoder.decode_y_pred(y1)
# decoder.visualize(score_boxes, x1, le=le, display=True)
# plt.show()

train_data = DataGenerator(train_df, config)
valid_data = DataGenerator(valid_df, config, mode='valid')
model = create_model(config, num_stacks=1)
model.summary()
print('Number os layers: %d' % len(model.layers))
train(model, train_data, valid_data, config)

# print(pred)
# print(pred.shape)


# myval = DataGenerator(valid_df, config, mode='valid')
# model = create_model(config)
# print(len(model.layers))
# model = 1
# current = calculate_map(config, model, myval)
# print('map', current)

# import tensorflow as tf

# print(tf.config.list_physical_devices('GPU'))