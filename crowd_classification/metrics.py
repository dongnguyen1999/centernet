from re import I
import os
from tqdm.std import trange
from utils.config import Config
from typing import List, Union
from tensorflow.keras.callbacks import Callback
import numpy as np
import cv2
import pandas as pd
import time

def calcScore(model, valid, config,  confidence=0.5, path=None, debug=False):
  
  tp, tn, fp, fn = 0, 0, 0, 0
  pred_time = 0

  for idx in trange(len(valid)):
    batchX, batchY = valid.__getitem__(idx)

    # if debug == True and y_true == 1: print(f'count image: {count_image(boxes, le)}; threshold: {crowd_threshold}; y_true: {y_true}')
    
    start_time = time.time()
    
    predY = model.predict(batchX)

    pred_time = (time.time() - start_time) * 1000

    for score_idx, score in enumerate(predY):
      y_pred = 1 if score > confidence else 0
      y_true = batchY[score_idx]
      # if debug == True and y_true == 1: print(f'score: {score}; confidence: {confidence}; y_pred: {y_pred}')

      if (y_true == 1 and y_pred == 1):
        tp += 1
      elif (y_true == 1 and y_pred == 0):
        fn += 1
      elif (y_true == 0 and y_pred == 1):
        fp += 1
      elif (y_true == 0 and y_pred == 0):
        tn += 1

  # if debug == True: print('tp, tn, fp, fn: ', tp, tn, fp, fn)
  accuracy = (tp + tn) / (tp + tn + fp + fn + 0.000001)
  precision = tp / (tp + fp + 0.000001)
  recall = tp / (tp + fn + 0.000001)
  f1 = (2*precision*recall) / (precision + recall + 0.000001)

  return accuracy, precision, recall, f1, pred_time



class SaveBestScore(Callback):
  def __init__(self, config: Config, path, valid_gen, confidence=0.5):
    super(SaveBestScore, self).__init__()
    self.config = config
    self.best_weights = None
    self.path = path
    self.confidence = confidence
    self.generator = valid_gen

  def on_train_begin(self, logs=None):
    self.best = 0
    self.best_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
    accuracy, prediction, recall, f1, _ = calcScore(self.model, self.generator, self.config, confidence=self.confidence)
    print('Valid current: acc %.4f, prec %.4f, rec %.4f, f1 %.4f' % (accuracy, prediction, recall, f1))
    if f1 > self.best:
      self.best = f1
      self.best_epoch = epoch
      self.best_weights = self.model.get_weights()
      print('Valid best f1 score: %.4f, saving model to %s' % (f1, os.path.join(self.path, '{epoch:03d}-{f1:.3f}.hdf5'.format(epoch=epoch, f1=f1))))
      self.model.save_weights(os.path.join(self.path, '{epoch:03d}-{f1:.3f}.hdf5'.format(epoch=epoch, f1=f1)))
  
  def on_train_end(self, logs=None):
    print('Training ended, the best f1 score weight is at epoch %03d with map %.3f' % (self.best_epoch, self.best))  

class TestScore(Callback):
  def __init__(self, config: Config, path, test_df, le, crowd_threshold, confidence=0.25):
    super(TestScore, self).__init__()
    self.config = config
    self.best_weights = None
    self.path = path
    self.confidence = confidence
    self.test_df = test_df
    self.crowd_threshold = crowd_threshold
    self.le = le
    self.test_paths = config.test_paths

  def on_epoch_end(self, epoch, logs=None):
    rdf = []
    columns = ['test_id', 'accuracy', 'prediction', 'recall', 'f1']
    for test_index, test_path in enumerate(self.test_paths):
      test_id = test_index + 1
      print(f'Evalutate test{test_id}')
      test_df = self.test_df[self.test_df['test_id'] == test_id]
      accuracy, prediction, recall, f1, _ = calcScore(self.model, test_df, self.le, self.crowd_threshold, self.config, confidence=self.confidence, path=test_path)
      print('test%d: acc %.4f, prec %.4f, rec %.4f, f1 %.4f' % (test_id, accuracy, prediction, recall, f1))
      r = [f'test{test_id}', accuracy, prediction, recall, f1]
      rdf.append(r)

    print('Saving...')
    rdf = pd.DataFrame(rdf, columns=columns)
    rdf.to_csv(os.path.join(self.path, 'map-epoch{epoch:02d}.csv'.format(epoch=epoch)), index=False, header=True)
  
