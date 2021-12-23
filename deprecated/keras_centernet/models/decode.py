
from utils.config import Config

from keras.models import Model
from keras import backend as K
from keras.layers import Lambda
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


def _nms(heat, kernel=3):
  hmax = K.pool2d(heat, (kernel, kernel), padding='same', pool_mode='max')
  keep = K.cast(K.equal(hmax, heat), K.floatx())
  return heat * keep


def _ctdet_decode(hm, k=100):
  hm = K.sigmoid(hm)
  hm = _nms(hm)
  hm_shape = K.shape(hm)
  batch, cat = hm_shape[0], hm_shape[3]

  hm_flat = K.reshape(hm, (batch, -1))

  def _process_sample(args):
    _hm = args[0]
    _scores, _inds = tf.math.top_k(_hm, k=k, sorted=True)
    _classes = K.cast(_inds % cat, 'float32')
    # _inds = K.cast(_inds / cat, 'int32')
    # _xs = K.cast(_inds % width, 'float32')
    # _ys = K.cast(K.cast(_inds / width, 'int32'), 'float32')

    _detection = K.stack([_scores, _classes], -1)
    return _detection

  detections = K.map_fn(_process_sample, [hm_flat], dtype=K.floatx())
  return detections


def CountDecode(model, k=100):
  def _decode(output):
    out = output[0]
    return _ctdet_decode(out, k=k)
  output = Lambda(_decode)(model.outputs)
  model = Model(model.input, output)
  return model


def visualize(box_and_score, img, config: Config, confidence=0.5, le=None, display=False):
  boxes = []
  scores = []
  color_scheme = [(0,0,255), (255,0,0), (0,255,255), (0,127,127), (127,255,127), (255,255,0)]
  number_of_rect = len(box_and_score)

  for i in range(number_of_rect):
    left, top, right, bottom, score, predicted_class = box_and_score[i, :]
    if (score > confidence):
      top = np.floor(top).astype('int32')
      left = np.floor(left).astype('int32')
      bottom = np.floor(bottom).astype('int32')
      right = np.floor(right).astype('int32')
      # top = np.floor(top * config.input_size / config.output_size).astype('int32')
      # left = np.floor(left * config.input_size / config.output_size).astype('int32')
      # bottom = np.floor(bottom * config.input_size / config.output_size).astype('int32')
      # right = np.floor(right * config.input_size / config.output_size).astype('int32')
      predicted_class = int(predicted_class)
      label = '{:.2f}'.format(score)
      if le != None:
          class_name = le.inverse_transform([predicted_class])[0]
          label = '{class_name} {label}'.format(class_name = class_name, label = label)

      #print(label)
      #print(top, left, right, bottom)
      cv2.rectangle(img, (left, top), (right, bottom), color_scheme[predicted_class], 2)
      cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX ,  
                  0.5, color_scheme[predicted_class], 2, cv2.LINE_AA) 
      boxes.append([left, top, right-left, bottom-top])
      scores.append(score)
      
  if display == True:
      fig, ax = plt.subplots(1, 1, figsize=(16, 8))
      ax.set_axis_off()
      ax.imshow(img)

  return np.array(boxes), np.array(scores)