from utils.config import Config
from utils.dataset.vn_vehicle import DataGenerator
from models.centernet import create_model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from utils.loss_functions import CenterNetLosses
from utils.callbacks import SaveBestmAP
import os
import numpy as np

#####TRAIN##########
    
def train(model, train_df, valid_df, config: Config):

    mygen = DataGenerator(train_df, config)
    myval = DataGenerator(valid_df, config, mode='valid')

    # EarlyStopping
    early_stopping = EarlyStopping(monitor = 'val_loss', min_delta=0, patience = 60, verbose = 1)

    # ModelCheckpoint

    if os.path.exists(config.checkpoint_path) == False: os.makedirs(config.checkpoint_path)

    frequently_save_path = os.path.join(config.checkpoint_path, 'every_3_epoch')
    if os.path.exists(frequently_save_path) == False: os.makedirs(frequently_save_path)
    model_frequently_checkpoint = ModelCheckpoint(os.path.join(frequently_save_path, "{epoch:02d}-{val_loss:.3f}.hdf5"), monitor = 'val_loss', verbose = 1,
                                         save_best_only = False, save_weights_only = True, period = 3)

    best_valloss_save_path = os.path.join(config.checkpoint_path, 'best_val_loss')
    if os.path.exists(best_valloss_save_path) == False: os.makedirs(best_valloss_save_path)
    model_bestloss_checkpoint = ModelCheckpoint(os.path.join(best_valloss_save_path, "{epoch:02d}-{val_loss:.3f}.hdf5"), monitor = 'val_loss', verbose = 1,
                                         save_best_only = True, save_weights_only = True, period = 1)

    best_map_save_path = os.path.join(config.checkpoint_path, 'best_map')
    if os.path.exists(best_map_save_path) == False: os.makedirs(best_map_save_path)
    save_best_map = SaveBestmAP(config, best_map_save_path, myval)


    # reduce learning rate
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.25, patience=2, min_lr=1e-5, verbose=1)

    centernet_losses = CenterNetLosses(config)
    centernet_loss, heatmap_loss, offset_loss, size_loss = centernet_losses.all_losses()
    model.compile(loss=centernet_loss, optimizer=Adam(learning_rate=config.lr), metrics=[heatmap_loss,size_loss,offset_loss])

    hist = model.fit(
        mygen,
        steps_per_epoch = len(train_df[config.image_id].unique()) // config.batch_size,
        epochs = config.epochs, 
        validation_data=myval,
        validation_steps = len(valid_df[config.image_id].unique()) // config.batch_size,
        callbacks = [early_stopping, reduce_lr, model_frequently_checkpoint, model_bestloss_checkpoint, save_best_map],
        shuffle = True,
        verbose = 1,
    )

    print(hist.history.keys())
    np.save(os.path.join(config.checkpoint_path, 'history.np') , hist.history)