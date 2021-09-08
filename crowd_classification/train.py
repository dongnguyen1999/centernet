from utils.config import Config
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import os
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from crowd_classification.losses import f1, precision, recall

def train(model, train, valid, test, config: Config):

    if config.weights_path != None:
        weights_path = config.weights_path
        if not (weights_path.startswith('/') or weights_path.startswith('C:')):
            weights_path = os.path.join(config.checkpoint_path, weights_path)
        # load weights
        print('Loading weights...')
        model.load_weights(weights_path, by_name=True)
        print('Done!')

    # ModelCheckpoint

    if os.path.exists(config.checkpoint_path) == False: os.makedirs(config.checkpoint_path)

    frequently_save_path = os.path.join(config.checkpoint_path, 'every_epoch')
    if os.path.exists(frequently_save_path) == False: os.makedirs(frequently_save_path)
    model_frequently_checkpoint = ModelCheckpoint(os.path.join(frequently_save_path, "{epoch:02d}-{val_loss:.3f}.hdf5"), monitor = 'val_loss', verbose = 1,
                                         save_best_only = False, save_weights_only = True, period = 5)

    best_valloss_save_path = os.path.join(config.checkpoint_path, 'best_val_loss')
    if os.path.exists(best_valloss_save_path) == False: os.makedirs(best_valloss_save_path)
    model_bestloss_checkpoint = ModelCheckpoint(os.path.join(best_valloss_save_path, "{epoch:02d}-{val_loss:.3f}.hdf5"), monitor = 'val_loss', verbose = 1,
                                         save_best_only = True, save_weights_only = True, period = 1)

    best_f1_save_path = os.path.join(config.checkpoint_path, 'best_f1')
    if os.path.exists(best_f1_save_path) == False: os.makedirs(best_f1_save_path)
    model_bestf1_checkpoint = ModelCheckpoint(os.path.join(best_f1_save_path, "{epoch:02d}-{f1:.3f}.hdf5"), monitor = 'val_f1', verbose = 1,
                                         save_best_only = True, save_weights_only = True, period = 1)
    
    # define logger to log loss and val_loss every epoch
    csv_logger = CSVLogger(os.path.join(config.checkpoint_path, "history_log.csv"), append=True)

    # reduce learning rate
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.25, patience=2, min_lr=1e-6, verbose=1)

    # compile model
    opt = SGD(lr=config.lr, momentum=config.momentum)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', precision, recall, f1])

    history = model.fit(train, steps_per_epoch=len(train), shuffle=True,
	    validation_data=valid, validation_steps=len(valid), epochs=config.epochs, verbose=1, 
        callbacks=[
            model_frequently_checkpoint,
            model_bestloss_checkpoint,
            model_bestf1_checkpoint,
            csv_logger,
            reduce_lr
        ])
    