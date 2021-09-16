from keras_centernet_count.models.decode import CountDecode
from re import I
from keras_centernet_count.utils import normalize_image
import os
from tqdm.std import trange
from utils.config import Config
from typing import List, Union
from tensorflow.keras.callbacks import Callback
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import mean_absolute_error


def calcMae(model, valid_df, config: Config, path=None):
  model_ = CountDecode(model)
  
  image_ids = valid_df[config.image_id].unique()

  y_true = []
  y_pred = []

  # for idx in trange(1):
  for idx in trange(len(image_ids)):
    image_id = image_ids[idx]
    img_name = os.path.basename(image_id)
    img_path = config.valid_path if path == None else path
    img = cv2.cvtColor(cv2.imread(os.path.join(img_path, img_name)), cv2.COLOR_BGR2RGB)
    im_h, im_w = img.shape[:2]
    img = normalize_image(img)
    img = cv2.resize(img, (config.input_size, config.input_size))

    boxes = valid_df[valid_df[config.image_id]==image_id]

    true_counts = []
    for i in range(config.num_classes):
      count = boxes[boxes['label'] == i].label.count()
      true_counts.append(count)

    true_counts = np.array(true_counts)
    y_true.append(true_counts)

    out = model_.predict(img[None])

    pred_counts = np.array([0 for _ in range(config.num_classes)])
    for detection in out[0]:
      count, label = detection
      pred_counts[int(label)] += count
    y_pred.append(pred_counts)

  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  maes = []
  for i in range(config.num_classes):
    maes.append(mean_absolute_error(y_true[:, i], y_pred[:, i]))
  maes = np.array(maes) #mae /class
  return maes

class SaveBestMae(Callback):
  def __init__(self, config: Config, path, valid_df):
    super(SaveBestMae, self).__init__()
    self.config = config
    self.best_weights = None
    self.path = path
    self.valid_df = valid_df
    self.num_classes = config.num_classes

  def on_train_begin(self, logs=None):
    self.best = np.inf
    self.best_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
    maes = calcMae(self.model, self.valid_df, self.config)
    current = np.sum(maes)
    print('Average MAE: %.4f' % current)
    for cls in range(self.num_classes):
      print('mae_class_%d: %.4f' % (cls, maes[cls]))

    if current < self.best:
      self.best = current
      self.best_epoch = epoch
      self.best_weights = self.model.get_weights()
      save_path = os.path.join(self.path, '{epoch:02d}-{mae:.3f}.hdf5'.format(epoch=epoch, mae=current))
      print('Best MAE: %.4f, saving model to %s' % (current, save_path))
      self.model.save_weights(save_path)
  
  def on_train_end(self, logs=None):
    print('Training ended, the best mae weight is at epoch %02d with mae %.3f' % (self.best_epoch, self.best))

class TestMae(Callback):
  def __init__(self, config: Config, path, valid_df, test_df):
    super(TestMae, self).__init__()
    self.config = config
    self.path = path
    self.test_df = test_df
    self.valid_df = valid_df
    self.valid_path = config.valid_path
    self.test_paths = config.test_paths
    self.num_classes = config.num_classes

  def on_epoch_end(self, epoch, logs=None):
    rdf = []
    columns = []

    print(f'Evalutate valid')
    maes = calcMae(self.model, self.valid_df, self.config, path=self.valid_path)
    current = np.sum(maes)
    r = [f'valid', current]
    names = ['test_id', 'average_mae']
    for cls in range(self.num_classes):
      print('mae_class_%d: %.4f' % (cls, maes[cls]))
      r.append(maes[cls])
      names.append('mae_class_%d' % cls)

    print('Valid average MAE: %.4f' % current)
    rdf.append(r)
    columns = names

    for test_index, test_path in enumerate(self.test_paths):
      test_id = test_index + 1
      print(f'Evalutate test{test_id}')
      test_df = self.test_df[self.test_df['test_id'] == test_id]
      maes = calcMae(self.model, test_df, self.config, path=test_path)
      current = np.sum(maes)
      r = [f'test{test_id}', current]

      for cls in range(self.num_classes):
        print('mae_class_%d: %.4f' % (cls, maes[cls]))
        r.append(maes[cls])

      print('test%s average MAE: %.4f' % (test_id, current))
      rdf.append(r)

    print('Saving...')
    rdf = pd.DataFrame(rdf, columns=columns)
    rdf.to_csv(os.path.join(self.path, 'mae-epoch{epoch:02d}.csv'.format(epoch=epoch)), index=False, header=True)
  
