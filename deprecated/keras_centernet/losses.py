import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from tensorflow import convert_to_tensor
from utils.config import Config
from tensorflow.keras.optimizers import Adam
from keras.losses import mean_squared_error

def compile_model(model, config: Config, alpha = 2., beta = 4.): # loss weights [hm, reg, wh]

  def _get_mask(y_true):
    mask = convert_to_tensor(np.zeros((config.output_size, config.output_size)).astype(np.float32))
    for category in range(config.num_classes):
      category_mask = K.sign(y_true[..., config.num_classes + category])
      mask = tf.add(mask, category_mask)
    return K.sign(mask)

  def heatmap_loss(y_true, y_pred):
    mask = _get_mask(y_true)
    N = K.sum(mask)
    y_pred = K.sigmoid(y_pred)

    heatmap_true = K.flatten(y_true[..., config.num_classes : 2*config.num_classes]) #Exact center points true
    heatmap_true_rate = K.flatten(y_true[..., : config.num_classes]) #Heatmap true
    heatmap_pred = K.flatten(y_pred[..., : config.num_classes]) #Predicted heatmap 

    heatloss = -K.sum(
        heatmap_true* ((1-heatmap_pred)**alpha)*K.log(heatmap_pred + 1e-6)
        + (1-heatmap_true)* ((1-heatmap_true_rate)**beta)*(heatmap_pred**alpha) * K.log(1-heatmap_pred + 1e-6)
    )

    return heatloss/N


  model.compile(loss=heatmap_loss, optimizer=Adam(learning_rate=config.lr), metrics=[heatmap_loss])
  return model