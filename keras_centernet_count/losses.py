import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from tensorflow import convert_to_tensor
from utils.config import Config
from tensorflow.keras.optimizers import Adam, SGD
from keras.losses import mean_squared_error

def compile_model(model, config: Config, alpha = 2., beta = 4.): # loss weights [hm, reg, wh]
  opt = SGD(lr = config.lr, momentum = 0.9, nesterov = True)
  model.compile(loss='mse', optimizer=opt)
  return model