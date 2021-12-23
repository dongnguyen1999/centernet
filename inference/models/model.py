import gc
import os
import time
import numpy as np

from inference.models.frame_difference import FrameDiffEstimator
from inference.utils import compute_count_score, get_masked_img, normalize_image
from inference.config import Config
# from matplotlib import pyplot as plt
# from tensorflow.keras.models import load_model
from inference.models.decode import CtDetDecode, CountDecode
from inference.models.cnn import create_model as create_crowd_model
from inference.models.hourglass import create_model as create_count_model
import cv2
import tensorflow as tf
from inference.utils import AsynTask
from tensorflow.keras.backend import clear_session

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

config = Config(3, None, None, None, None, input_size=512)

ROOT_PATH = os.getcwd()
# print(ROOT_PATH)
WEIGHTS_PATH = os.path.join(ROOT_PATH, 'inference', 'weights')

model_configs = {
    'crowd_model': {
        'architecture': 'ClsVgg16',
        'weights': os.path.join(WEIGHTS_PATH, 'vgg16_fineturning_epoch2.hdf5')
    },
    'crowd_model_focalbin': {
        'architecture': 'ClsVgg16',
        'weights': os.path.join(WEIGHTS_PATH, 'vgg16_fineturning_focalbin_epoch10.hdf5')
    },
    'count_model_1stack': {
        'architecture': 'HmOnlyHourglass1Stack',
        'weights': os.path.join(WEIGHTS_PATH, 'count_hg1stack_epoch6.hdf5')
    },
    'count_model_2stack': {
        'architecture': 'HmOnlyHourglass2Stack',
        'weights': os.path.join(WEIGHTS_PATH, 'count_hg2stack_epoch6.hdf5')
    },
    'detect_model': {
        'architecture': 'DtHourglass1Stack',
        'weights': os.path.join(WEIGHTS_PATH, 'detect_hg1stack_tfmosaic_epoch6.hdf5')
    },
}

default_config = {
    'count_conf_threshold': 0.5,
    'classify_conf_threshold': 0.5,
    'crowd_thresholds': [10, 25],
    'count_thresholds': [10, 20],
    'show_bounding_box': True,
    'show_diff_mask': False,
    'show_foreground_mask': False,
    'reset_session': False,
    'aspect_ratio': 1,
}


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


def load_models(crowd_model_config, count_model_config):
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
    def __init__(self, crowd_model_name, count_model_name, debug=False):

        global default_config

        self.current_pred_time = 0
        self.current_classify_time = 0
        self.current_count_time = 0
        self.current_bs_time = 0

        self.debug = debug

        self.make_model(crowd_model_name, count_model_name)
        self.apply_editable_config(default_config)

        self.reset_session()

    def make_model(self, crowd_model_name, count_model_name):
        global model_configs

        self.crowd_model = None
        self.count_model = None

        gc.collect()

        crowd_model_config = model_configs[crowd_model_name]
        count_model_config = model_configs[count_model_name]

        crowd_model, count_model = load_models(crowd_model_config, count_model_config)

        self.crowd_model_name = crowd_model_name

        self.count_model_name = count_model_name

        self.frame_diff_estimator = FrameDiffEstimator()

        self.heatmap_only = 'HmOnly' in count_model_config['architecture']

        self.crowd_model = crowd_model

        self.count_model = count_model

        self.reset_session()

    def reset_session(self):
        self.frame_diff_estimator.refresh()

        self.pred_time = 0
        self.count_time = 0
        self.bs_time = 0
        self.classify_time = 0

        self.pred_count = 0
        self.count_count = 0
        self.bs_count = 0

        self.process_cache = []

    '''
        Editable config:
        default_config = {
            'count_conf_threshold': 0.5,
            'classify_conf_threshold': 0.5,
            'crowd_thresholds': [5, 20],
            'count_thresholds': [10, 20],
            'show_bounding_box': True,
            'show_diff_mask': False,
            'show_foreground_mask': False,
        }
    '''

    def apply_editable_config(self, config):
        self.count_thresholds = config['count_thresholds']
        self.crowd_thresholds = config['crowd_thresholds']
        self.count_conf_threshold = config['count_conf_threshold']
        self.classify_conf_threshold = config['classify_conf_threshold']

    def average_pred_time(self):
        return self.pred_time / self.pred_count

    def average_classify_time(self):
        return self.classify_time / self.pred_count

    def average_count_time(self):
        return self.count_time / self.count_count

    def average_bs_time(self):
        return self.bs_time / self.bs_count

    @AsynTask
    def classify(self, image):

        pre_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pre_image = pre_image / 255

        cls_time = time.time()

        if self.crowd_model is None:
            return
        pred = self.crowd_model.predict(pre_image[None])

        classify_time = (time.time() - cls_time) * 1000
        self.current_classify_time = classify_time
        self.classify_time += classify_time

        self.process_cache.append({
            'image': image,
            'cls_score': pred[0]
        })

    def predict(self, image, mask=None, config=default_config):
        start_time = time.time()

        self.apply_editable_config(config)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image = cv2.resize(image, (512, 512))

        handle = self.classify(image)

        if len(self.process_cache) > 0:
            cls_output = self.process_cache.pop(0)
            image = cls_output['image']
            score = cls_output['cls_score']

            if score >= self.classify_conf_threshold:

                # 1 - crowd street
                bs_start_time = time.time()
                estimation = self.frame_diff_estimator.apply(image, mask=mask, return_diff_mask=config['show_diff_mask'], return_foreground_mask=config['show_foreground_mask'])

                bs_time = (time.time() - bs_start_time) * 1000
                self.current_bs_time = bs_time
                self.bs_time += bs_time
                self.bs_count += 1

                first_pivot, second_pivot = self.crowd_thresholds
                diff_rate = estimation['diff_rate']

                if diff_rate < first_pivot:
                    estimation['label'] = 5
                if first_pivot <= diff_rate < second_pivot:
                    estimation['label'] = 4
                if diff_rate >= second_pivot:
                    estimation['label'] = 3

            else:
                # 0 - normal street
                estimation = {}
                if config['show_diff_mask'] or config['show_foreground_mask'] or self.debug:
                    estimation = self.frame_diff_estimator.apply(image, mask=mask, return_diff_mask=config['show_diff_mask'], return_foreground_mask=config['show_foreground_mask'])
                else:
                    self.frame_diff_estimator.feed(image, mask=mask)

                count_time_start = time.time()

                pre_image = normalize_image(image)

                if mask is not None:
                    pre_image = get_masked_img(pre_image, mask)
                    if self.debug:
                        cv2.putText(pre_image, 'Diff rate: %.3f' % estimation['diff_rate'], (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1, cv2.LINE_AA)
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

                estimation['detections'] = detections

                first_pivot, second_pivot = self.count_thresholds
                if count < first_pivot:
                    estimation['label'] = 0
                if first_pivot <= count < second_pivot:
                    estimation['label'] = 1
                if count >= second_pivot:
                    estimation['label'] = 2

            pred_time = (time.time() - start_time) * 1000
            self.pred_time += pred_time
            self.current_pred_time = pred_time

            self.pred_count += 1

            return image, estimation
        else:
            handle.join()
            return None
