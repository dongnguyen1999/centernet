from keras_centernet_cls.metrics import SaveBestScore, TestScore
from keras_centernet_cls.losses import compile_model
from keras_centernet_cls.dataset.vn_vehicle import DataGenerator
from utils.config import Config
from models.centernet import create_model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import CSVLogger
from utils.loss_functions import CenterNetLosses
# from keras_centernet_cls.metrics import SaveBestmAP, TestmAP
import os
import numpy as np

#####TRAIN##########
    
def train(model, train_df, valid_df, le, crowd_threshold, config: Config, test_df=None, generator=DataGenerator):

    train_data = generator(train_df, le, crowd_threshold, config)
    valid_data = generator(valid_df, le, crowd_threshold, config, mode='valid')

    if config.weights_path != None:
        weights_path = config.weights_path
        if not (weights_path.startswith('/') or weights_path.startswith('C:')):
            weights_path = os.path.join(config.checkpoint_path, weights_path)
        # load weights
        print('Loading weights...')
        model.load_weights(weights_path, by_name=True)
        print('Done!')

    # EarlyStopping
    early_stopping = EarlyStopping(monitor = 'val_loss', min_delta=0, patience = 60, verbose = 1)

    # ModelCheckpoint

    if os.path.exists(config.checkpoint_path) == False: os.makedirs(config.checkpoint_path)

    frequently_save_path = os.path.join(config.checkpoint_path, 'every_epoch')
    if os.path.exists(frequently_save_path) == False: os.makedirs(frequently_save_path)
    model_frequently_checkpoint = ModelCheckpoint(os.path.join(frequently_save_path, "{epoch:02d}-{val_loss:.3f}.hdf5"), monitor = 'val_loss', verbose = 1,
                                         save_best_only = False, save_weights_only = True, period = 1)

    best_valloss_save_path = os.path.join(config.checkpoint_path, 'best_val_loss')
    if os.path.exists(best_valloss_save_path) == False: os.makedirs(best_valloss_save_path)
    model_bestloss_checkpoint = ModelCheckpoint(os.path.join(best_valloss_save_path, "{epoch:02d}-{val_loss:.3f}.hdf5"), monitor = 'val_loss', verbose = 1,
                                         save_best_only = True, save_weights_only = True, period = 1)

    best_f1_save_path = os.path.join(config.checkpoint_path, 'best_f1')
    if os.path.exists(best_f1_save_path) == False: os.makedirs(best_f1_save_path)
    save_best_map = SaveBestScore(config, best_f1_save_path, valid_df, le, crowd_threshold)
    
    # define logger to log loss and val_loss every epoch
    csv_logger = CSVLogger(os.path.join(config.checkpoint_path, "history_log.csv"), append=True)

    # reduce learning rate
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.25, patience=2, min_lr=1e-6, verbose=1)

    callbacks = [early_stopping, reduce_lr, model_frequently_checkpoint, model_bestloss_checkpoint, save_best_map, csv_logger]

    if test_df is not None:
        test_score_save_path = os.path.join(config.checkpoint_path, 'test_score')
        if os.path.exists(test_score_save_path) == False: os.makedirs(test_score_save_path)
        save_test_score = TestScore(config, test_score_save_path, test_df, le, crowd_threshold)
        callbacks.append(save_test_score)

    model = compile_model(model, config)
    
    hist = model.fit(
        train_data,
        steps_per_epoch = len(train_data),
        epochs = config.epochs, 
        validation_data=valid_data,
        validation_steps = len(valid_data),
        callbacks = callbacks,
        shuffle = True,
        verbose = 1,
    )
