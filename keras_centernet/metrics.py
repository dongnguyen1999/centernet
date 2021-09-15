from keras_centernet.models.decode import CountDecode
from re import I
from keras_centernet.utils import normalize_image
import os
from tqdm.std import trange
from utils.config import Config
from typing import List, Union
from tensorflow.keras.callbacks import Callback
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import mean_absolute_error


def calcMae(model, valid_df, config: Config, confidences=[0.25, 0.5, 0.7], path=None):
  model_ = CountDecode(model)
  
  image_ids = valid_df[config.image_id].unique()

  y_true = []
  y_preds = [[] for _ in confidences]

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

    for conf_idx, confidence in enumerate(confidences):
      pred_counts = np.array([0,0,0])
      for detection in out[0]:
        conf, label = detection
        if conf > confidence:
          pred_counts[int(label)] += 1
      y_preds[conf_idx].append(pred_counts)

  y_true = np.array(y_true)
  y_preds = [np.array(y_pred) for y_pred in y_preds]

  conf_maes = []
  for y_pred in y_preds:
    class_maes = []
    for i in range(config.num_classes):
      class_maes.append(mean_absolute_error(y_true[:, i], y_pred[:, i]))
    conf_maes.append(class_maes)
  conf_maes = np.array(conf_maes) #mae /conf /class
  return conf_maes

class SaveBestMae(Callback):
  def __init__(self, config: Config, path, valid_df, confidences=[0.25, 0.5, 0.7]):
    super(SaveBestMae, self).__init__()
    self.config = config
    self.best_weights = None
    self.path = path
    self.confidences = confidences
    self.valid_df = valid_df

  def on_train_begin(self, logs=None):
    self.best = np.inf
    self.best_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
    conf_maes = calcMae(self.model, self.valid_df, self.config, confidences=self.confidences)
    sum_maes = np.sum(conf_maes, axis=1)
    for index, conf in enumerate(self.confidences):
      print('mae@%.2f: %.4f' % (conf, sum_maes[index]))
    current = np.mean(sum_maes)
    print('Average MAE: %.4f' % current)

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
  def __init__(self, config: Config, path, valid_df, test_df, confidences=[0.25, 0.5, 0.7]):
    super(TestMae, self).__init__()
    self.config = config
    self.path = path
    self.confidences = confidences
    self.test_df = test_df
    self.valid_df = valid_df
    self.valid_path = config.valid_path
    self.test_paths = config.test_paths
    self.num_classes = config.num_classes

  def on_epoch_end(self, epoch, logs=None):
    rdf = []
    columns = []

    print(f'Evalutate valid')
    conf_maes = calcMae(self.model, self.valid_df, self.config, path=self.valid_path, confidences=self.confidences)
    sum_maes = np.sum(conf_maes, axis=1)
    current = np.mean(sum_maes)
    r = [f'valid', current]
    names = ['test_id', 'average_mae']
    for index, conf in enumerate(self.confidences):
      print('mae@%.2f: %.4f' % (conf, sum_maes[index]))
      r.append(sum_maes[index])
      names.append('mae@%.2f' % conf)

      for cls in range(self.num_classes):
        print('mae@%.2f_class_%d: %.4f' % (conf, cls, conf_maes[index, cls]))
        r.append(conf_maes[index, cls])
        names.append('mae@%.2f_class_%d' % (conf, cls))

    print('Valid average MAE: %.4f' % current)
    rdf.append(r)
    columns = names

    for test_index, test_path in enumerate(self.test_paths):
      test_id = test_index + 1
      print(f'Evalutate test{test_id}')
      test_df = self.test_df[self.test_df['test_id'] == test_id]
      conf_maes = calcMae(self.model, test_df, self.config, path=test_path, confidences=self.confidences)
      sum_maes = np.sum(conf_maes, axis=1)
      current = np.mean(sum_maes)
      r = [f'test{test_id}', current]
      for index, conf in enumerate(self.confidences):
        print('mae@%.2f: %.4f' % (conf, sum_maes[index]))
        r.append(sum_maes[index])

        for cls in range(self.num_classes):
          print('mae@%.2f_class_%d: %.4f' % (conf, cls, conf_maes[index, cls]))
          r.append(conf_maes[index, cls])

      print('test%s average MAE: %.4f' % (test_id, current))
      rdf.append(r)

    print('Saving...')
    rdf = pd.DataFrame(rdf, columns=columns)
    rdf.to_csv(os.path.join(self.path, 'mae-epoch{epoch:02d}.csv'.format(epoch=epoch)), index=False, header=True)
  
