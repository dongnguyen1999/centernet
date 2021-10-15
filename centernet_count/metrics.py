from re import I
from centernet_count.utils import normalize_image
import os
from tqdm.std import trange
from utils.config import Config
from centernet_count.models.decode import CountDecode
from typing import List, Union
from tensorflow.keras.callbacks import Callback
import numpy as np
import cv2
import pandas as pd

def calculate_iou(gt, pred) -> float:
  """Calculates the IoU.
  """
  xt1, yt1, xt2, yt2, lt = gt
  xp1, yp1, xp2, yp2, lp = pred

  if lt != lp: return 0.0

  overlap_area = 0.0
  union_area = 0.0

  # Calculate overlap area
  dx = min(xt2, xp2) - max(xt1, xp1)
  dy = min(yt2, yp2) - max(yt1, yp1)

  if (dx > 0) and (dy > 0):
    overlap_area = dx * dy

  # Calculate union area
  union_area = (
    (xt2 - xt1) * (yt2 - yt1) +
    (xp2 - xp1) * (yp2 - yp1) -
    overlap_area
  )

  return overlap_area / union_area

def find_best_match(gts, predd, threshold=0.5):
  """Returns the index of the 'best match' between the
  ground-truth boxes and the prediction. The 'best match'
  is the highest IoU. (0.0 IoUs are ignored).
  
  Args:
    gts: Coordinates of the available ground-truth boxes
    pred: Coordinates of the predicted box
    threshold: Threshold
    form: Format of the coordinates
    
  Return:
    Index of the best match GT box (-1 if no match above threshold)
  """
  best_match_iou = -np.inf
  best_match_idx = -1
  
  for gt_idx, ggt in enumerate(gts):
    iou = calculate_iou(ggt, predd)
    
    if iou < threshold:
      continue
    
    if iou > best_match_iou:
      best_match_iou = iou
      best_match_idx = gt_idx

  return best_match_idx

def calculate_precision(preds_sorted, gt_boxes, num_classes, threshold=0.5):
  """Calculates precision per at one threshold.
  
  Args:
    preds_sorted: 
  """
  tp = 0
  fp = 0
  fn = 0

  cl_tp, cl_fn, cl_fp = np.array([0 for _ in range(num_classes)]), np.array([0 for _ in range(num_classes)]), np.array([0 for _ in range(num_classes)])

  fp_boxes = []

  for pred_idx, pred in enumerate(preds_sorted):
    best_match_gt_idx = find_best_match(gt_boxes, pred, threshold=threshold)

    if best_match_gt_idx >= 0:
      # True positive: The predicted box matches a gt box with an IoU above the threshold.
      tp += 1
      _, _, _, _, label = gt_boxes[best_match_gt_idx]
      cl_tp[label] += 1
      # Remove the matched GT box
      gt_boxes = np.delete(gt_boxes, best_match_gt_idx, axis=0)

    else:
      # No match
      # False positive: indicates a predicted box had no associated gt box.
      fp += 1
      _, _, _, _, label = pred
      cl_fp[label] += 1
      fp_boxes.append(pred)

  # False negative: indicates a gt box had no associated predicted box.
  fn = len(gt_boxes)
  for gtb in gt_boxes:
    _, _, _, _, label = gtb
    cl_fn[label] += 1

  precision = tp / (tp + fp + fn + 0.0000001)
  cl_precision = cl_tp / ((cl_tp + cl_fp + cl_fn) + 0.0000001)
  return precision, cl_precision, fp_boxes, gt_boxes

def calculate_image_precision(preds_sorted, gt_boxes, num_classes, thresholds=(0.5), debug=False):
  
  n_threshold = len(thresholds)
  image_precision = 0.0
  threshold_precision = []

  image_cl_precision = np.array([0.0 for _ in range(num_classes)])
  threshold_cl_precision = []
  
  for threshold in thresholds:
    precision_at_threshold, cl_precision_at_threshold, _, _ = calculate_precision(preds_sorted,
                               gt_boxes, num_classes,
                               threshold=threshold
                            )
    if debug:
      print("@{0:.2f} = {1:.4f}".format(threshold, precision_at_threshold))

    threshold_precision.append(precision_at_threshold)
    threshold_cl_precision.append(cl_precision_at_threshold)
    image_precision += precision_at_threshold / n_threshold
    image_cl_precision += cl_precision_at_threshold / n_threshold
  
  return image_precision, np.array(threshold_precision), image_cl_precision, np.array(threshold_cl_precision)

def calcMae(model, valid_df, config: Config, confidence=0.5, path=None):
  model_ = CountDecode(model)
  image_ids = valid_df[config.image_id].unique()

  sum_cls_maes = np.array([0.0 for _ in range(config.num_classes)])
  sum_mae = 0

  true_counts = np.array([0.0 for _ in range(config.num_classes)])
  sum_count = 0

  cls_N = np.array([0.000001 for _ in range(config.num_classes)])
  N = len(image_ids)
  
  for idx in trange(len(image_ids)):
    image_id = image_ids[idx]
    img_name = os.path.basename(image_id)
    img_path = config.valid_path if path == None else path
    
    img = cv2.imread(os.path.join(img_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_h, im_w = img.shape[:2]

    img = normalize_image(img)
    img = cv2.resize(img, (config.input_size, config.input_size))

    boxes = valid_df[valid_df[config.image_id]==image_id][['x1', 'y1', 'x2', 'y2', 'label']].values

    true_count = np.array([0.0 for _ in range(config.num_classes)])

    for box in boxes:
      x1, y1, x2, y2, label = box
      true_count[int(label)] += 1
    
    true_counts += true_count
    sum_count += np.sum(true_count)

    for i in range(config.num_classes):
      if true_count[i] > 0:
        cls_N[i] += 1

    out = model_.predict(img[None])

    pred_count = np.array([0.0 for _ in range(config.num_classes)])
    for detection in out[0]:
      conf, label = detection
      if conf > confidence:
        pred_count[int(label)] += 1

    cls_maes = np.abs(pred_count - true_count)
    mae = np.abs(np.sum(pred_count) - np.sum(true_count))

    sum_cls_maes += cls_maes
    sum_mae += mae

  sum_cls_maes /= cls_N
  sum_mae /= N

  true_counts /= cls_N
  sum_count /= N

  return sum_mae, sum_cls_maes

class SaveBestMae(Callback):
  def __init__(self, config: Config, path, valid_df, confidence=0.5):
    super(SaveBestMae, self).__init__()
    self.config = config
    self.best_weights = None
    self.path = path
    self.confidence = confidence
    self.valid_df = valid_df

  def on_train_begin(self, logs=None):
    self.best = 999999
    self.best_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
    current, _ = calcMae(self.model, self.valid_df, self.config, confidence=self.confidence)
    print('Current MAE: %.4f' % current)
    if current < self.best:
      self.best = current
      self.best_epoch = epoch+1
      self.best_weights = self.model.get_weights()
      print('Best MAE: %.4f, saving model to %s' % (current, os.path.join(self.path, '{epoch:02d}-{mae:.3f}.hdf5'.format(epoch=epoch+1, mae=current))))
      self.model.save_weights(os.path.join(self.path, '{epoch:02d}-{mae:.3f}.hdf5'.format(epoch=epoch+1, mae=current)))
  
  def on_train_end(self, logs=None):
    print('Training ended, the best mae weight is at epoch %02d with mae %.3f' % (self.best_epoch, self.best))

class TestMae(Callback):
  def __init__(self, config: Config, path, valid_df, test_df, confidence=0.5):
    super(TestMae, self).__init__()
    self.config = config
    self.path = path
    self.confidence = confidence
    self.test_df = test_df
    self.valid_df = valid_df
    self.test_paths = config.test_paths
    self.num_classes = config.num_classes
    self.df = pd.DataFrame([])

  def on_epoch_end(self, epoch, logs=None):
    rdf = []
    columns = []
    thresholds = [x for x in self.thresholds]
    
    print(f'Evalutate valid')
    current, maes = calcMae(self.model, self.valid_df, self.config, confidence=self.confidence)
    r = [f'valid', epoch+1, current]
    names = ['test_id', 'epoch', 'MAE']

    for cl in range(self.num_classes):
      print('MAE_class%d: %.4f' % (cl, maes[cl]))
      r.append(maes[cl])
      names.append('MAE_class%d' % cl)
      
    print('valid MAE: %.4f' % current)
    rdf.append(r)
    columns = names

    for test_index, test_path in enumerate(self.test_paths):
      test_id = test_index + 1
      print(f'Evalutate test{test_id}')
      test_df = self.test_df[self.test_df['test_id'] == test_id]
      current, maes = calcMae(self.model, test_df, self.config, path=test_path, confidence=self.confidence)
      r = [f'test{test_id}', epoch+1, current]

      for cl in range(self.num_classes):
        print('MAE_class%d: %.4f' % (cl, maes[cl]))
        r.append(maes[cl])
          
      print('test%s MAE: %.4f' % (test_id, current))
      rdf.append(r)

    print('Saving...')
    rdf = pd.DataFrame(rdf, columns=columns)
    rdf.to_csv(os.path.join(self.path, 'map-epoch{epoch:02d}.csv'.format(epoch=epoch+1)), index=False, header=True)
    if len(self.df) == 0:
      self.df = rdf
    else:
      self.df = pd.concat([self.df, rdf])

  def on_train_end(self, logs=None):
    print('Training ended, Saving result')  
    self.df.to_csv(os.path.join(self.path, 'map-test.csv'), index=False, header=True)
