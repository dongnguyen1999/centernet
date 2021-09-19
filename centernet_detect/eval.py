from utils.map_utils.calc_map import calc_map
from centernet_detect.utils import normalize_image
import cv2
from tqdm.std import trange
from centernet_detect.models.decode import CtDetDecode
from centernet_detect.losses import compile_model
from centernet_detect.dataset.vn_vehicle import DataGenerator
from utils.config import Config
import pandas as pd
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import CSVLogger
from centernet_detect.metrics import SaveBestmAP, TestmAP
import os
import numpy as np
from glob import glob
import shutil

#####TRAIN##########

def eval_models(valid_df, test_df, config: Config, model_prefix=None, eval_category='every_epoch', model_ckpt_paths=[], model_garden={}, confidence=0.2, threshold=0.5):

    if model_prefix != None:
        model_ckpt_paths = glob(os.path.join(config.logging_base, 'models', f'{model_prefix}*/'))
    # print(model_ckpt_paths)
    for ckpt_path in model_ckpt_paths:
        model_name = os.path.basename(ckpt_path[:-1])
        for k in model_garden:
            if k in model_name:
                print(f'Creating {k} model for weights in {model_name}')
                model = model_garden[k]
                break
        versions = glob(os.path.join(ckpt_path, 'v*/'))
        for version in versions:
            version_name = os.path.basename(version[:-1])
            version_model_name = f'{model_name}_{version_name}'
            print(f'Evaluating model {version_model_name}')
            eval_model(model, os.path.join(version, eval_category), valid_df, test_df, config, model_name=f'{version_model_name}_{eval_category}', confidence=confidence, threshold=threshold)
    
    
        
    
def eval_model(model, checkpoint_path, valid_df, test_df, config: Config, confidence=0.25, threshold=0.5, model_name=None):

    ckpt_weights_files = glob(os.path.join(checkpoint_path, '*.hdf5'))

    for ckpt_file in ckpt_weights_files:
        model.load_weights(ckpt_file)
        ckpt_filename = os.path.basename(ckpt_file)
        epoch_num = int(ckpt_filename[: ckpt_filename.find('-')])

        print(f'Epoch {epoch_num}: Evalutate valid')
        eval(model, f'{model_name}_epoch{epoch_num}', valid_df, 'valid', config, confidence=confidence, iou_threshold=threshold)

        for test_index, test_path in enumerate(config.test_paths):
            test_id = test_index + 1
            print(f'Epoch {epoch_num}: Evalutate test{test_id}')
            current_test_df = test_df[test_df['test_id'] == test_id]
            eval(model, f'{model_name}_epoch{epoch_num}', current_test_df, f'test{test_id}', config, confidence=confidence, iou_threshold=threshold)

    
def eval(model, model_name, test_df, testset_name, config: Config, confidence=0.2, iou_threshold=0.5, test_path=None):
    model_ = CtDetDecode(model)
    image_ids = test_df[config.image_id].unique()
    save_path = os.path.join(config.logging_base, 'eval', f'{model_name}_{testset_name}')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    gt_path = os.path.join(save_path, 'input', 'ground-truth')
    dt_path = os.path.join(save_path, 'input', 'detection-results')
    tem_img_path = os.path.join(config.logging_base, 'eval', testset_name)
    save_tem_img = False
    if not os.path.exists(tem_img_path):
        os.makedirs(tem_img_path)
        save_tem_img = True
    if not os.path.exists(dt_path):
        os.makedirs(dt_path)
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    
    for idx in trange(len(image_ids)):
        detections = []
        ground_truths = []

        image_id = image_ids[idx]
        img_name = os.path.basename(image_id)
        img_path = config.valid_path if test_path == None else test_path
        img = cv2.cvtColor(cv2.imread(os.path.join(img_path, img_name)), cv2.COLOR_BGR2RGB)
        im_h, im_w = img.shape[:2]

        if save_tem_img:
            cv2.imwrite(os.path.join(tem_img_path, img_name), cv2.resize(img, (config.input_size, config.input_size)))

        img = normalize_image(img)
        img = cv2.resize(img, (config.input_size, config.input_size))

        boxes = test_df[test_df[config.image_id]==image_id]

        boxes.x1 = np.floor(boxes.x1 * config.input_size / im_w)
        boxes.y1 = np.floor(boxes.y1 * config.input_size / im_h)
        boxes.x2 = np.floor(boxes.x2 * config.input_size / im_w)
        boxes.y2 = np.floor(boxes.y2 * config.input_size / im_h)

        boxes = boxes[['x1', 'y1', 'x2', 'y2', 'label']].values
        boxes = boxes.astype('int32')

        out = model_.predict(img[None])

        for detection in out[0]:
            x1, y1, x2, y2, conf, label = detection
            if conf > confidence:
                detections.append([int(label), conf, x1, y1, x2, y2])
        
        for box in boxes:
            x1, y1, x2, y2, label = box
            ground_truths.append([int(label), x1, y1, x2, y2])

        detections = pd.DataFrame(detections)
        ground_truths = pd.DataFrame(ground_truths)

        detections.to_csv(os.path.join(dt_path, f'{img_name[:-4]}.txt'), index=False, header=False, sep=' ')
        ground_truths.to_csv(os.path.join(gt_path, f'{img_name[:-4]}.txt'), index=False, header=False, sep=' ')
        # print(boxes, pred_box)

    calc_map(save_path, tem_img_path, iou_threshold)
  
    
