import os
import time
import numpy as np
from inference.utils import compute_count_score, get_masked_img, normalize_image
from utils.config import Config 
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from inference.models.decode import CtDetDecode, CountDecode
from inference.models.cnn import create_model as create_crowd_model
from inference.models.hourglass import create_model as create_count_model
import cv2

config = Config(3, None, None, None, None, input_size=512)

def create_clsvgg16_model(weights):
    global config
    model = create_crowd_model(config, architecture="pretrained_vgg16", freeze_feature_block=False)
    model.load_weights(weights)
    return model

def create_hmonlyhourglass1stack_model(weights):
    global config
    model = create_count_model(config, num_stacks=1, heatmap_only=True)
    model.load_weights(weights)
    model = CountDecode(model)
    return model

def create_hmonlyhourglass2stack_model(weights):
    global config
    model = create_count_model(config, num_stacks=2, heatmap_only=True)
    model.load_weights(weights)
    model = CountDecode(model)
    return model

def create_dthourglass1stack_model(weights):
    global config
    model = create_count_model(config, num_stacks=1)
    model.load_weights(weights)
    model = CtDetDecode(model)
    return model

model_garden = {
    'ClsVgg16': create_clsvgg16_model,
    'HmOnlyHourglass1Stack': create_hmonlyhourglass1stack_model,
    'HmOnlyHourglass2Stack': create_hmonlyhourglass2stack_model,
    'DtHourglass1Stack': create_dthourglass1stack_model
}

def create_models(crowd_model_config, count_model_config):
    crowd_weights = crowd_model_config['weights']
    count_weights = count_model_config['weights']

    if not os.path.exists(crowd_weights) or not os.path.exists(count_weights):
        raise ValueError("Not found weights!")    

    CrowdModel = model_garden[crowd_model_config['architecture']]
    crowd_model = CrowdModel(crowd_weights)

    CountModel = model_garden[count_model_config['architecture']]
    count_model = CountModel(count_weights)
    
    return crowd_model, count_model

class Model:
    def __init__(self, crowd_model, count_model, frame_diff_estimator, crowd_thresholds=[10, 25], count_thresholds=[0.5, 2.0], classify_conf_threshold=0, count_conf_threshold=0.25, heatmap_only=False, debug=False):
        self.crowd_model = crowd_model
        self.count_model = count_model
        self.frame_diff_estimator = frame_diff_estimator
        self.count_thresholds = count_thresholds
        self.crowd_thresholds = crowd_thresholds

        self.current_pred_time = 0
        self.current_classify_time = 0
        self.current_count_time = 0
        self.current_bs_time = 0

        self.pred_time = 0
        self.count_time = 0
        self.classify_time = 0
        self.bs_time = 0

        self.pred_count = 0
        self.bs_count = 0
        self.count_count = 0

        self.count_conf_threshold = count_conf_threshold
        self.classify_conf_threshold = classify_conf_threshold

        self.heatmap_only = heatmap_only
        self.debug = debug

    def reset_session(self):
        self.frame_diff_estimator.refresh()

        self.pred_time = 0
        self.detect_time = 0
        self.bs_time = 0
        self.classify_time = 0

        self.pred_count = 0
        self.detect_count = 0
        self.bs_count = 0
    
    '''
        Editable config:
        {
            count_conf_threshold: 0.5,
            classify_conf_threshold: 0.5,
            count_thresholds: [10, 20],
            crowd_thresholds: [5, 20],
        }
    '''
    def apply_editable_config(self, config):
        self.count_thresholds = config.count_thresholds
        self.crowd_thresholds = config.crowd_thresholds
        self.count_conf_threshold = config.count_conf_threshold
        self.classify_conf_threshold = config.classify_conf_threshold

    def average_pred_time(self):
        return self.pred_time / self.pred_count

    def average_classify_time(self):
        return self.classify_time / self.pred_count

    def average_count_time(self):
        return self.count_time / self.count_count
    
    def average_bs_time(self):
        return self.bs_time / self.bs_count

    def predict(self, image, mask=None):

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.current_classify_time = 0
        self.current_count_time = 0
        self.current_pred_time = 0
        self.current_bs_time = 0

        image = cv2.resize(image, (512, 512))
        start_time = time.time()

        pre_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pre_image = pre_image / 255
        predY = self.crowd_model.predict(pre_image[None])

        classify_time = (time.time() - start_time) * 1000
        self.current_classify_time = classify_time
        self.classify_time += classify_time

        score = predY[0]
        if score >= self.classify_conf_threshold:

            # 1 - crowd street
            bs_start_time = time.time()
            diff_rate = self.frame_diff_estimator.apply(image, mask=mask)

            bs_time = (time.time() - bs_start_time) * 1000
            self.current_bs_time = bs_time
            self.bs_time += bs_time
            self.bs_count += 1

            first_pivot, second_pivot = self.crowd_thresholds
            if diff_rate < first_pivot:
                result = (5, diff_rate)
            if diff_rate >= first_pivot and diff_rate < second_pivot:
                result = (4, diff_rate)
            if diff_rate >= second_pivot:
                result = (3, diff_rate)

        else:
            # 0 - normal street
            if self.debug == True:
                diff_rate = self.frame_diff_estimator.apply(image, mask=mask)
            else:
                self.frame_diff_estimator.feed(image, mask=mask)

            count_time_start = time.time()

            pre_image = normalize_image(image)
            
            if mask is not None:
                pre_image = get_masked_img(pre_image, mask)
                if self.debug == True:
                    cv2.putText(pre_image, 'Diff rate: %.3f' % diff_rate, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA) 
                    cv2.imshow('count_model.pre_image', pre_image)

            out = self.count_model.predict(pre_image[None])
            
            count_time = (time.time() - count_time_start) * 1000
            self.current_count_time = count_time
            self.count_time += count_time
            self.count_count += 1

            score_idx = 0 if self.heatmap_only == True else 4
            label_idx = 1 if self.heatmap_only == True else 5
            detections = out[0]
            detections = detections[detections[:, score_idx] > self.count_conf_threshold]
            count = compute_count_score((
                np.size(detections[detections[:, label_idx] == 0], axis=0),
                np.size(detections[detections[:, label_idx] == 1], axis=0),
                np.size(detections[detections[:, label_idx] == 2], axis=0)
            ))

            first_pivot, second_pivot = self.count_thresholds
            if count < first_pivot:
                result = (0, detections)
            if count >= first_pivot and count < second_pivot:
                result = (1, detections)
            if count >= second_pivot:
                result = (2, detections)
            
        pred_time = (time.time() - start_time) * 1000
        self.pred_time += pred_time
        self.current_pred_time = pred_time

        self.pred_count += 1

        return result




        
