import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from tensorflow import convert_to_tensor
from utils.config import Config
from tensorflow.keras.optimizers import Adam, SGD
from keras.losses import mean_squared_error

def compile_model(model, config: Config, loss_weights=[1, 1, 0.1], alpha = 2., beta = 4.): # loss weights [hm, reg, wh]

  def size_loss(y_true, y_pred):
    mask = y_true[..., 2]
    N = K.sum(mask)
    sizeloss = K.sum(
        K.abs(y_true[..., 0] * mask - y_pred[..., 0] * mask)
        + K.abs(y_true[..., 1] * mask - y_pred[..., 1] * mask)
    )
    return sizeloss/N

  def offset_loss(y_true, y_pred):
    mask = y_true[..., 2]
    N = K.sum(mask)
    offsetloss = K.sum(
      K.abs(y_true[..., 0] * mask - y_pred[..., 0] * mask)
      + K.abs(y_true[..., 1] * mask - y_pred[..., 1] * mask)
    )
    return offsetloss/N

  def heatmap_loss(y_true, y_pred):
    mask = y_true[..., 2*config.num_classes]
    y_pred = K.sigmoid(y_pred)
    N=K.sum(mask)

    heatmap_true = K.flatten(y_true[..., config.num_classes : 2*config.num_classes]) #Exact center points true
    heatmap_true_rate = K.flatten(y_true[..., : config.num_classes]) #Heatmap true
    heatmap_pred = K.flatten(y_pred[..., : config.num_classes]) #Predicted heatmap 

    heatloss = -K.sum(
        heatmap_true* ((1-heatmap_pred)**alpha)*K.log(heatmap_pred + 1e-6)
        + (1-heatmap_true)* ((1-heatmap_true_rate)**beta)*(heatmap_pred**alpha) * K.log(1-heatmap_pred + 1e-6)
    )

    return heatloss/N
    
  # opt = Adam(learning_rate=config.lr)
  opt = SGD(learning_rate=config.lr, momentum=config.momentum)
  model.compile(loss=[heatmap_loss, offset_loss, size_loss], optimizer=opt, loss_weights=loss_weights)
  return model