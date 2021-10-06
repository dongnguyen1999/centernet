import os
import time
import numpy as np
from centernet_detect.utils import normalize_image
from inference.model.decode import CtDetDecode
from utils.config import Config 
from inference.model.cnn import create_model as create_classify_model
from inference.model.hourglass import create_model as create_detect_model
from matplotlib import pyplot as plt
import cv2

def create_models(crowd_weight, detect_weight):
    config = Config(3, None, None, None, None, input_size=512)
    if not os.path.exists(crowd_weight) or not os.path.exists(detect_weight):
        raise ValueError("Not found weights!")
    crowd_model = create_classify_model(config, architecture="pretrained_vgg16", freeze_feature_block=False)
    crowd_model.load_weights(crowd_weight)
    detect_model = create_detect_model(config, num_stacks=1)
    detect_model.load_weights(detect_weight)
    detect_model = CtDetDecode(detect_model)

    return crowd_model, detect_model

    


class Model:
    def __init__(self, crowd_model, detect_model, background_subtractor, classify_threshold, detect_threshold, count_thresholds, crowd_thresholds):
        self.crowd_model = crowd_model
        self.detect_model = detect_model
        self.background_subtractor = background_subtractor
        self.count_thresholds = count_thresholds
        self.crowd_thresholds = crowd_thresholds
        self.pred_time = 0
        self.detect_time = 0
        self.classify_time = 0
        self.pred_count = 0
        self.detect_threshold = detect_threshold
        self.classify_threshold = classify_threshold
    

    def reset_session(self):
        self.background_subtractor.refresh()
        self.pred_time = 0
        self.detect_time = 0
        self.classify_time = 0
        self.pred_count = 0

    def average_pred_time(self):
        return self.pred_time / self.pred_count

    def average_classify_time(self):
        return self.classify_time / self.pred_count

    def average_detect_time(self):
        return self.detect_time / self.pred_count

    def predict(self, image):
        image = cv2.resize(image, (512, 512))
        start_time = time.time()
        self.pred_count += 1

        predY = self.crowd_model.predict(image[None])

        classify_time = (time.time() - start_time) * 1000
        self.classify_time += classify_time

        score = predY[0]
        if score >= self.classify_threshold:
            mask, diff_rate = self.background_subtractor.apply(image, return_diff_rate=True)
            cv2.imshow("Mask", mask)
            first_pivot, second_pivot = self.crowd_thresholds
            if diff_rate < first_pivot:
                result = (3, None)
            if diff_rate >= first_pivot and diff_rate < second_pivot:
                result = (4, None)
            if diff_rate >= second_pivot:
                result = (5, None)
        else:
            mask = self.background_subtractor.apply(image)
            cv2.imshow("Mask", mask)
            detect_time_start = time.time()

            pre_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            pre_image = normalize_image(pre_image)

            out = self.detect_model.predict(pre_image[None])
            
            detect_time = (time.time() - detect_time_start) * 1000
            self.detect_time += detect_time
            detections = out[0]
            detections = detections[detections[:, 4] > self.detect_threshold]
            count = 0
            count += np.size(detections[detections[:, 5] == 0], axis=0)
            count += np.size(detections[detections[:, 5] == 1], axis=0) * 2
            count += np.size(detections[detections[:, 5] == 2], axis=0) * 2                
            first_pivot, second_pivot = self.count_thresholds
            if count < first_pivot:
                result = (0, detections)
            if count >= first_pivot and count < second_pivot:
                result = (1, detections)
            if count >= second_pivot:
                result = (2, detections)
            
        pred_time = (time.time() - start_time) * 1000
        self.pred_time = pred_time

        return result




        
