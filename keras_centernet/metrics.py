from re import I
from keras_centernet.utils import normalize_image
import os
from tqdm.std import trange
from utils.config import Config
from keras_centernet.models.decode import CtDetDecode
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

def calculate_precision(preds_sorted, gt_boxes, threshold=0.5):
	"""Calculates precision per at one threshold.
	
	Args:
		preds_sorted: 
	"""
	tp = 0
	fp = 0
	fn = 0

	fp_boxes = []

	for pred_idx, pred in enumerate(preds_sorted):
		best_match_gt_idx = find_best_match(gt_boxes, pred, threshold=threshold)

		if best_match_gt_idx >= 0:
			# True positive: The predicted box matches a gt box with an IoU above the threshold.
			tp += 1

			# Remove the matched GT box
			gt_boxes = np.delete(gt_boxes, best_match_gt_idx, axis=0)

		else:
			# No match
			# False positive: indicates a predicted box had no associated gt box.
			fp += 1
			fp_boxes.append(pred)

	# False negative: indicates a gt box had no associated predicted box.
	fn = len(gt_boxes)
	precision = tp / (tp + fp + fn)
	return precision, fp_boxes, gt_boxes

def calculate_image_precision(preds_sorted, gt_boxes, thresholds=(0.5), debug=False):
	
	n_threshold = len(thresholds)
	image_precision = 0.0
	threshold_precision = []
	
	for threshold in thresholds:
		precision_at_threshold, _, _ = calculate_precision(preds_sorted,
														   gt_boxes,
														   threshold=threshold
														)
		if debug:
			print("@{0:.2f} = {1:.4f}".format(threshold, precision_at_threshold))
		threshold_precision.append(precision_at_threshold)
		image_precision += precision_at_threshold / n_threshold
	
	return image_precision, np.array(threshold_precision)

def calcmAP(model, valid_df, config: Config, confidence=0.5, thresholds=np.arange(0.5, 0.76, 0.05), path=None):
  model_ = CtDetDecode(model, config.num_classes)
  
  iou_thresholds = [x for x in thresholds]
  
  precision = []
  threshold_precisions = np.array([0.0 for _ in iou_thresholds])

  image_ids = valid_df[config.image_id].unique()
  countN = 0
  for idx in trange(len(image_ids)):
    image_id = image_ids[idx]
    img_name = os.path.basename(image_id)
    img_path = config.valid_path if path == None else path
    img = cv2.cvtColor(cv2.imread(os.path.join(img_path, img_name)), cv2.COLOR_BGR2RGB)
    img = normalize_image(img)
    img = cv2.resize(img, (config.input_size, config.input_size))

    boxes = valid_df[valid_df[config.image_id]==image_id][['x1', 'y1', 'x2', 'y2', 'label']].values

    out = model_.predict(img[None])
    pred_box,scores=[],[]

    for detection in out[0]:
      x1, y1, x2, y2, conf, label = detection
      if conf > confidence:
        pred_box.append([x1, y1, x2, y2, label])
        scores.append(conf)

    pred_box = np.array(pred_box, dtype=np.int32)
    scores = np.array(scores)

    preds_sorted_idx = np.argsort(scores)[::-1]
    preds_sorted = pred_box[preds_sorted_idx]

    if len(boxes) > 0:
      image_precision, threshold_precision = calculate_image_precision(preds_sorted, boxes,
                                              thresholds=iou_thresholds, debug=False)
      precision.append(image_precision)
      threshold_precisions += threshold_precision
      countN += 1
    else:
      if len(preds_sorted) > 0:
        precision.append(0)
  
  precision = np.array(precision)
  return np.mean(precision), (threshold_precisions / countN)

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
    current, maps = calcmAP(self.model, self.valid_df, self.config, confidence=self.confidence, thresholds=self.thresholds)
    for index, iou in enumerate(self.thresholds):
      print('mAP@%.2f: %.4f' % (iou, maps[index]))
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
  def __init__(self, config: Config, path, test_df, confidence=0.25, thresholds=np.arange(0.5, 0.76, 0.05)):
    super(TestmAP, self).__init__()
    self.config = config
    self.path = path
    self.thresholds = thresholds
    self.confidence = confidence
    self.test_df = test_df
    self.test_paths = config.test_paths

  def on_epoch_end(self, epoch, logs=None):
    rdf = []
    columns = []
    for test_index, test_path in enumerate(self.test_paths):
      test_id = test_index + 1
      print(f'Evalutate test{test_id}')
      test_df = self.test_df[self.test_df['test_id'] == test_id]
      current, maps = calcmAP(self.model, test_df, self.config, path=test_path, confidence=self.confidence, thresholds=self.thresholds)
      r = [f'test{test_id}', current]
      names = ['test_id', 'mAP']
      for index, iou in enumerate(self.thresholds):
        print('mAP@%.2f: %.4f' % (iou, maps[index]))
        r.append(maps[index])
        names.append('mAP@%.2f' % iou)
      print('test%s mAP: %.4f' % (test_id, current))
      rdf.append(r)
      columns = names

    print('Saving...')
    rdf = pd.DataFrame(rdf, columns=columns)
    rdf.to_csv(os.path.join(self.path, 'map-epoch{epoch:02d}.csv'.format(epoch=epoch)), index=False, header=True)
  
