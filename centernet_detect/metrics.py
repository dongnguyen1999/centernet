from re import I
from centernet_detect.utils import normalize_image
import os
from tqdm.std import trange
from utils.config import Config
from centernet_detect.models.decode import CtDetDecode
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

def calcmAP(model, valid_df, config: Config, confidence=0.5, thresholds=np.arange(0.5, 0.76, 0.05), path=None):
  model_ = CtDetDecode(model)
  
  iou_thresholds = [x for x in thresholds]
  
  precision = []
  cl_precision = []
  threshold_precisions = np.array([0.0 for _ in iou_thresholds])
  threshold_cl_precisions = np.zeros((len(iou_thresholds), config.num_classes))

  image_ids = valid_df[config.image_id].unique()
  countN = 0
  for idx in trange(len(image_ids)):
    image_id = image_ids[idx]
    img_name = os.path.basename(image_id)
    img_path = config.valid_path if path == None else path
    img = cv2.imread(os.path.join(img_path, img_name))
    
    im_h, im_w = img.shape[:2]
    img = normalize_image(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (config.input_size, config.input_size))

    boxes = valid_df[valid_df[config.image_id]==image_id]

    boxes.x1 = np.floor(boxes.x1 * config.input_size / im_w)
    boxes.y1 = np.floor(boxes.y1 * config.input_size / im_h)
    boxes.x2 = np.floor(boxes.x2 * config.input_size / im_w)
    boxes.y2 = np.floor(boxes.y2 * config.input_size / im_h)

    boxes = boxes[['x1', 'y1', 'x2', 'y2', 'label']].values
    boxes = boxes.astype('int32')

    out = model_.predict(img[None])
    pred_box,scores=[],[]

    for detection in out[0]:
      x1, y1, x2, y2, conf, label = detection
      if conf > confidence:
        pred_box.append([x1, y1, x2, y2, label])
        scores.append(conf)

    pred_box = np.array(pred_box, dtype=np.int32)
    scores = np.array(scores)

    # print(boxes, pred_box)

    preds_sorted_idx = np.argsort(scores)[::-1]
    preds_sorted = pred_box[preds_sorted_idx]

    if len(boxes) > 0:
      image_precision, threshold_precision, image_cl_precision, threshold_cl_precision = calculate_image_precision(preds_sorted, boxes, config.num_classes,
                                              thresholds=iou_thresholds, debug=False)
      precision.append(image_precision)
      cl_precision.append(image_cl_precision)
      threshold_precisions += threshold_precision
      threshold_cl_precisions += threshold_cl_precision
      countN += 1
    else:
      if len(preds_sorted) > 0:
        precision.append(0)
  
  precision = np.array(precision)
  cl_precision = np.array(cl_precision)
  return np.mean(precision), (threshold_precisions / countN), np.mean(cl_precision, axis=0), (threshold_cl_precisions / countN)

class SaveBestmAP(Callback):
  def __init__(self, config: Config, path, valid_df, confidence=0.25, thresholds=np.arange(0.5, 0.76, 0.05)):
    super(SaveBestmAP, self).__init__()
    self.config = config
    self.best_weights = None
    self.path = path
    self.thresholds = thresholds
    self.confidence = confidence
    self.valid_df = valid_df

  def on_train_begin(self, logs=None):
    self.best = 0
    self.best_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
    current, _, _, _ = calcmAP(self.model, self.valid_df, self.config, confidence=self.confidence, thresholds=self.thresholds)
    print('Current mAP: %.4f' % current)
    if np.greater(current, self.best):
      self.best = current
      self.best_epoch = epoch
      self.best_weights = self.model.get_weights()
      print('Best mAP: %.4f, saving model to %s' % (current, os.path.join(self.path, '{epoch:02d}-{map:.3f}.hdf5'.format(epoch=epoch, map=current))))
      self.model.save_weights(os.path.join(self.path, '{epoch:02d}-{map:.3f}.hdf5'.format(epoch=epoch, map=current)))
  
  def on_train_end(self, logs=None):
    print('Training ended, the best map weight is at epoch %02d with map %.3f' % (self.best_epoch, self.best))

class TestmAP(Callback):
  def __init__(self, config: Config, path, valid_df, test_df, confidence=0.25, thresholds=np.arange(0.5, 0.76, 0.05)):
    super(TestmAP, self).__init__()
    self.config = config
    self.path = path
    self.thresholds = thresholds
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
    current, maps, cl_maps, th_cl_maps = calcmAP(self.model, self.valid_df, self.config, confidence=self.confidence, thresholds=self.thresholds)
    r = [f'valid', epoch, current]
    names = ['test_id', 'epoch', 'mAP']
    for index, iou in enumerate(thresholds):
      # print('mAP@%.2f: %.4f' % (iou, maps[index]))
      r.append(maps[index])
      names.append('mAP@%.2f' % iou)

    for cl in range(self.num_classes):
      print('mAP_class%d: %.4f' % (cl, cl_maps[cl]))
      r.append(cl_maps[cl])
      names.append('mAP_class%d' % cl)
      for index, iou in enumerate(thresholds):
        # print('mAP@%.2f_class%d: %.4f' % (iou, cl, th_cl_maps[index, cl]))
        r.append(th_cl_maps[index, cl])
        names.append('mAP@%.2f_class%d' % (iou, cl))
      
    print('valid mAP: %.4f' % current)
    rdf.append(r)
    columns = names

    for test_index, test_path in enumerate(self.test_paths):
      test_id = test_index + 1
      print(f'Evalutate test{test_id}')
      test_df = self.test_df[self.test_df['test_id'] == test_id]
      current, maps, cl_maps, th_cl_maps = calcmAP(self.model, test_df, self.config, path=test_path, confidence=self.confidence, thresholds=self.thresholds)
      r = [f'test{test_id}', epoch, current]
      for index, iou in enumerate(thresholds):
        # print('mAP@%.2f: %.4f' % (iou, maps[index]))
        r.append(maps[index])

      for cl in range(self.num_classes):
        print('mAP_class%d: %.4f' % (cl, cl_maps[cl]))
        r.append(cl_maps[cl])
        for index, iou in enumerate(thresholds):
          # print('mAP@%.2f_class%d: %.4f' % (iou, cl, th_cl_maps[index, cl]))
          r.append(th_cl_maps[index, cl])
          
      print('test%s mAP: %.4f' % (test_id, current))
      rdf.append(r)

    print('Saving...')
    rdf = pd.DataFrame(rdf, columns=columns)
    rdf.to_csv(os.path.join(self.path, 'map-epoch{epoch:02d}.csv'.format(epoch=epoch)), index=False, header=True)
    if len(self.df) == 0:
      self.df = rdf
    else:
      self.df = pd.concat([self.df, rdf])

  def on_train_end(self, logs=None):
    print('Training ended, Saving result')  
    self.df.to_csv(os.path.join(self.path, 'map-test.csv'), index=False, header=True)
