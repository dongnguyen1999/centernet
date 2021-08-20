from collections import namedtuple
from typing import List, Union
from utils.config import Config
from utils.output_decoder import OutputDecoder
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import cv2

# def show_image(config, x, y):
#     for category in range(config.num_classes):
#         points = np.argwhere(y[:,:, config.num_classes+4 + category] == 1)

#         for y1,x1 in points:
#             # print(y1,x1)
#             offsety = y[:,:, config.num_classes + 0][y1,x1]
#             offetx = y[:,:, config.num_classes + 1][y1,x1]
#             h = y[:,:, config.num_classes + 2][y1,x1] * config.input_size/4
#             w = y[:,:, config.num_classes + 3][y1,x1] * config.input_size/4

#             x1, y1 = x1+offetx, y1+offsety 

#             xmin = int((x1-w/2)*4)
#             xmax = int((x1+w/2)*4)
#             ymin = int((y1-h/2)*4)
#             ymax = int((y1+h/2)*4)

#             cv2.rectangle(x, (xmin, ymin), (xmax, ymax), (0,255,255), 2)
#             cv2.circle(x, (int(x1*4),int(y1*4)), 2, (255,0,0), -1) 

#     #cv2.imshow('djpg',y[:,:,1]*255)
#     #cv2.imshow('drawjpg',x)
#     fig, ax = plt.subplots(1, 1, figsize=(16, 8))

#     ax.set_axis_off()
#     ax.imshow(x)
#     plt.show()

def calculate_iou(true_box, pred_box) -> float:
    #[category, score, top, left, bottom, right]
    #                  ymin, xmin, ymax, xmax

    # print('truebox', true_box.shape)
    true_category, true_score, true_top, true_left, true_bottom, true_right = true_box
    pred_category, pred_score, pred_top, pred_left, pred_bottom, pred_right = pred_box

    if true_category != pred_category: return 0.0

    overlap_area = 0.0
    union_area = 0.0

    # Calculate overlap area
    dx = min(true_right, pred_right) - max(true_left, pred_left)
    dy = min(true_bottom, pred_bottom) - max(true_top, pred_top)

    if (dx > 0) and (dy > 0):
        overlap_area = dx * dy
    # Calculate union area
    union_area = (
        (true_right - true_left)*(true_bottom - true_top) +
        (pred_right - pred_left)*(pred_bottom - pred_top) -
        overlap_area
    )
    return overlap_area / union_area

def find_best_match(true_boxes, pred_box, threshold=0.5):
    best_match_iou = -np.inf
    best_match_idx = -1
    
    for index, true_box in enumerate(true_boxes):
        iou = calculate_iou(true_box, pred_box)
        
        if iou < threshold:
            continue
        
        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = index

    return best_match_idx


def calculate_precision(pred_boxes, true_boxes, threshold=0.5):
    tp = 0
    fp = 0
    fn = 0

    fp_boxes = []

    for index, pred_box in enumerate(pred_boxes):
        best_true_match_idx = find_best_match(true_boxes, pred_box, threshold=threshold)

        if best_true_match_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            true_boxes = np.delete(true_boxes, best_true_match_idx, axis=0)

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1
            fp_boxes.append(pred_box)

    # False negative: indicates a gt box had no associated predicted box.
    fn = len(true_boxes)
    precision = tp / (tp + fp + fn)
    return precision, fp_boxes, true_boxes


def calculate_map(config: Config, model, valid_generator, threshold=0.5):
    precisions = []
    output_decoder = OutputDecoder(config, score_threshold=threshold)

    for count, (X, y_true) in enumerate(valid_generator):
        batch_ground_truths = [output_decoder.decode_y_true(y) for y in y_true]
        # show_image(config ,X[0], y_true[0])

        y_pred = model.predict(X)

        # y_pred = y_true[..., :7]
        batch_score_boxes = [output_decoder.decode_y_pred(y) for y in y_pred]

        for (true_boxes, pred_boxes) in zip(batch_ground_truths, batch_score_boxes):
            if np.size(true_boxes) == 0 or np.size(pred_boxes) == 0:
                precision = 0.0
            else: precision, _, _ = calculate_precision(pred_boxes, true_boxes, threshold=threshold)
            # print(precision)
            precisions.append(precision)
            
    precisions = np.array(precisions)
    return np.mean(precisions)
