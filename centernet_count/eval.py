from utils.map_utils.calc_map import calc_map, read_map_output
from centernet_count.utils import normalize_image
import cv2
from tqdm.std import trange
from centernet_count.losses import compile_model
from centernet_count.dataset.vn_vehicle import DataGenerator
from utils.config import Config
import pandas as pd
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import CSVLogger
from centernet_count.models.decode import CountDecode
# from centernet_count.metrics import SaveBestmAP, TestmAP
import os
import numpy as np
from glob import glob
import shutil
from datetime import datetime
import time

#####TRAIN##########

def eval_models(valid_df, test_df, config: Config, model_prefix=None, eval_category='every_epoch', model_ckpt_paths=[], model_garden={}, confidence=0, threshold=0.5, metric='map'):
    eval_result = []
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
            if metric == 'map':
                eval_model(model, os.path.join(version, eval_category), valid_df, test_df, config, model_name=f'{version_model_name}_{eval_category}', confidence=confidence, threshold=threshold)
            elif metric == 'mae':
                result = eval_model(model, os.path.join(version, eval_category), valid_df, test_df, config, model_name=f'{version_model_name}_{eval_category}', confidence=confidence, threshold=threshold, metric='mae')
                eval_result.extend(result)

    if metric == 'mae':
        ts = datetime.now()
        eval_result = pd.DataFrame(eval_result, columns=['model_name', 'testset', 'conf', 'mae', 'mae_2w', 'mae_4w', 'mae_prio', 'time'])
        eval_result.to_csv(os.path.join(config.logging_base, 'eval', f'{model_prefix}_{ts.timestamp()}.csv'), index=False, header=True)

    
    
        
    
def eval_model(model, checkpoint_path, valid_df, test_df, config: Config, confidence, threshold, model_name=None, metric='map'):

    ckpt_weights_files = glob(os.path.join(checkpoint_path, '*.hdf5'))
    model_result = []

    for ckpt_file in ckpt_weights_files:
        model.load_weights(ckpt_file)
        ckpt_filename = os.path.basename(ckpt_file)
        epoch_num = int(ckpt_filename[: ckpt_filename.find('-')])

        print(f'Epoch {epoch_num}: Evalutate valid')
        if metric == 'map':
            eval(model, f'{model_name}_epoch{epoch_num}', valid_df, 'valid', config, confidence=confidence, iou_threshold=threshold)
        elif metric == 'mae':
            result = eval_mae(model, f'{model_name}_epoch{epoch_num}', valid_df, 'valid', config, confidence=confidence, iou_threshold=threshold)
            model_result.append(result)

        for test_index, test_path in enumerate(config.test_paths):
            test_id = test_index + 1
            print(f'Epoch {epoch_num}: Evalutate test{test_id}')
            current_test_df = test_df[test_df['test_id'] == test_id]
            if metric == 'map':
                eval(model, f'{model_name}_epoch{epoch_num}', current_test_df, f'test{test_id}', config, confidence=confidence, iou_threshold=threshold, test_path=test_path)
            elif metric == 'mae':
                result = eval_mae(model, f'{model_name}_epoch{epoch_num}', current_test_df, f'test{test_id}', config, confidence=confidence, iou_threshold=threshold, test_path=test_path)
                model_result.append(result)
    
    return model_result

    
def eval(model, model_name, test_df, testset_name, config: Config, confidence, iou_threshold, test_path=None):
    model_ = CountDecode(model)
    image_ids = test_df[config.image_id].unique()
    save_path = os.path.join(config.logging_base, 'eval', f'{model_name}_{testset_name}_conf{confidence}_iou{iou_threshold}')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    gt_path = os.path.join(save_path, 'input', 'ground-truth')
    dt_path = os.path.join(save_path, 'input', 'detection-results')
    tem_img_path = os.path.join(config.data_base, 'eval', testset_name)
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
        
        img = cv2.imread(os.path.join(img_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

    calc_map(save_path, tem_img_path, iou_threshold, temp_path=os.path.join(config.data_base, 'eval', '.temp_files'))

def eval_mae(model, model_name, test_df, testset_name, config: Config, confidence, iou_threshold, test_path=None):
    model_ = CountDecode(model)
    image_ids = test_df[config.image_id].unique()

    sum_cls_maes = np.array([0.0 for _ in range(config.num_classes)])
    sum_mae = 0
    true_counts = np.array([0.0 for _ in range(config.num_classes)])
    sum_count = 0
    cls_N = np.array([0.000001 for _ in range(config.num_classes)])
    N = len(image_ids)
    runtime = 0
    
    for idx in trange(len(image_ids)):
        detections = []
        ground_truths = []

        image_id = image_ids[idx]
        img_name = os.path.basename(image_id)
        img_path = config.valid_path if test_path == None else test_path
        
        img = cv2.imread(os.path.join(img_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_h, im_w = img.shape[:2]

        img = normalize_image(img)
        img = cv2.resize(img, (config.input_size, config.input_size))

        boxes = test_df[test_df[config.image_id]==image_id][['x1', 'y1', 'x2', 'y2', 'label']].values

        true_count = np.array([0.0 for _ in range(config.num_classes)])

        for box in boxes:
            x1, y1, x2, y2, label = box
            true_count[int(label)] += 1
        
        true_counts += true_count
        sum_count += np.sum(true_count)

        for i in range(config.num_classes):
            if true_count[i] > 0:
                cls_N[i] += 1

        start_time = time.time()
        out = model_.predict(img[None])
        runtime += (time.time() - start_time) * 1000

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

    print(f'Average counting {testset_name}: {sum_count}; {true_counts.tolist()}')

    return_array = [model_name, testset_name, confidence, sum_mae]
    return_array.extend(sum_cls_maes.tolist())
    return_array.append(runtime/N)
    print(return_array)
    return return_array


def read_eval_output(model_prefix, config: Config):
    df = []
    for eval_result in glob(os.path.join(config.logging_base, 'eval', f'{model_prefix}*')):
        map, aps = read_map_output(eval_result, config.num_classes)
        eval_name = os.path.basename(eval_result)
        epoch = eval_name[eval_name.find('epoch')+5: eval_name.find('_', eval_name.find('epoch'))]
        testset = 'valid' if eval_name.find('valid') != -1 else eval_name[eval_name.find('test'): eval_name.find('_', eval_name.find('test'))]
        iou = eval_name[eval_name.find('iou')+3:]
        df.append([eval_name, epoch, testset, iou, map, aps[0], aps[1], aps[2]])
        print(eval_name, epoch, testset, iou, map, aps)
    df = pd.DataFrame(df, columns=['model_name', 'epoch', 'testset', 'iou', 'map', 'ap0', 'ap1', 'ap2'])
    df.to_csv(os.path.join(config.logging_base, f'{model_prefix}_eval.csv'), index=False)




    
