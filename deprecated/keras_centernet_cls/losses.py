import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from tensorflow import convert_to_tensor
from utils.config import Config
from tensorflow.keras.optimizers import Adam
from keras.losses import mean_squared_error

def compile_model(model, config: Config, loss_weights=[1, 1], alpha = 2., beta = 4.): # loss weights [hm, reg, wh]

  def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

  def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    _precision = true_positives / (predicted_positives + K.epsilon())    
    return _precision

  def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    _recall = true_positives / (possible_positives + K.epsilon())
    return _recall

  model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=config.lr), metrics=['accuracy', precision, recall, f1])
  return model