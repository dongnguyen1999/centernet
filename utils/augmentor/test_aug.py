from utils.augmentor.misc import MiscEffect, multi_scale
from utils.augmentor.color import VisualEffect
from tqdm.std import trange
from utils.config import Config
import os
import cv2
import numpy as np

def visualize_test(df, img_path, config: Config, limit=None):
    image_ids = df[config.image_id].unique()

    visual_effect = VisualEffect()
    misc_effect = MiscEffect()

    lim = limit if limit is not None else len(image_ids)
    for idx in trange(lim):
        image_id = image_ids[idx]
        img_name = os.path.basename(image_id)
        img = cv2.imread(os.path.join(img_path, img_name))

        src_boxes = df[df[config.image_id]==image_id]
        
        boxes = src_boxes[['x1', 'y1', 'x2', 'y2']].values
        print(boxes)

        for box in boxes.astype(np.int32):
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        src_image = img.copy()
        # cv2.namedWindow('src_image', cv2.WINDOW_NORMAL)
        cv2.imshow('src_image', src_image)

        # img = visual_effect(img)
        img, boxes = misc_effect(img, boxes)
        # img, boxes = multi_scale(img, boxes)
        print(np.hstack((boxes, src_boxes[['label']].values)))

        img = img.copy()

        for box in boxes.astype(np.int32):
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)

        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)
        cv2.waitKey(0)


       