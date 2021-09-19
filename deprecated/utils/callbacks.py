from utils.config import Config
from tensorflow.keras.callbacks import Callback
import numpy as np
from utils.metrics import calculate_map
import os


class SaveBestmAP(Callback):
  def __init__(self, config: Config, path, valid_generator, threshold=0.5):
    super(SaveBestmAP, self).__init__()
    self.config = config
    self.best_weights = None
    self.path = path
    self.threshold = threshold
    self.valid_generator = valid_generator

  def on_train_begin(self, logs=None):
    self.best = 0
    self.best_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
    current = calculate_map(self.config, self.model, self.valid_generator, self.threshold)
    if np.greater(current, self.best):
      self.best = current
      self.best_epoch = epoch
      self.best_weights = self.model.get_weights()
      print(f'Best mAP: {current}, saving...')
      self.model.save_weights(os.path.join(self.path, '{epoch:02d}-{map:.3f}.hdf5'.format(epoch=epoch, map=current)))
    else:
      print(f'Current mAP: {current}')
  
  def on_train_end(self, logs=None):
    print('Training ended, the best map weight is at epoch %02d with map %.3f' % (self.best_epoch, self.best))

