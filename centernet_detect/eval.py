from centernet_detect.metrics import calcmAP
from keras_centernet_count.metrics import calcMaeV0
from keras_centernet_count.losses import compile_model
from keras_centernet_count.dataset.vn_vehicle import DataGenerator
from utils.config import Config
from models.centernet import create_model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import CSVLogger
from utils.loss_functions import CenterNetLosses
from glob import glob
import os
# from keras_centernet_count.metrics import SaveBestmAP, TestmAP
import os
import numpy as np
import pandas as pd

#####TRAIN##########

def eval_models(valid_df, test_df, config: Config, model_prefix=None, model_ckpt_paths=[], model_garden={}):

    if model_prefix != None:
        model_ckpt_paths = glob(os.path.join(config.logging_base, f'models/{model_prefix}*/'))
    # print(model_ckpt_paths)
    result = pd.DataFrame([], columns=['model_name', 'epoch', 'testset', 'mAP',	'mAP@0.50',	'AP@0.50_class0', 'AP@0.50_class1', 'AP@0.50_class2', 'mAP@0.70', 'AP@0.70_class0', 'AP@0.70_class1', 'AP@0.70_class2'])
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
            df = eval(model, os.path.join(version, 'every_epoch'), valid_df, test_df, config, model_name=version_model_name)
            result = pd.concat([result, df])
    
    result.to_csv(os.path.join(config.logging_base, f'eval_{model_prefix}.csv'), index=False, header=True)
    
        
    
def eval(model, checkpoint_path, valid_df, test_df, config: Config, confidence=0.2, thresholds=(0.5, 0.7), generator=DataGenerator, model_name=None):

    df = []

    ckpt_weights_files = glob(os.path.join(checkpoint_path, '*.hdf5'))
    if model_name == None:
        model_name = checkpoint_path

    for ckpt_file in ckpt_weights_files:
        model.load_weights(ckpt_file)
        ckpt_filename = os.path.basename(ckpt_file)
        epoch_num = int(ckpt_filename[: ckpt_filename.find('-')])
        
        print(f'Epoch {epoch_num}: Evalutate valid')

        current, th_maps, th_cl_maps = calcmAP(model, valid_df, config, confidence=confidence, thresholds=thresholds)
        r = [model_name, epoch_num, f'valid', current]

        for index, iou in enumerate(thresholds):
            print('mAP@%.2f: %.4f' % (iou, th_maps[index]))
            r.append(th_maps[index])

        ms = ""
        for cl in range(config.num_classes):
            ms += 'AP@%.2f_class%d: %.4f; ' % (iou, cl, th_cl_maps[index, cl])
            r.append(th_cl_maps[index, cl])
        print(ms)
        
        
        print('valid mAP: %.4f' % current)
        df.append(r)

        for test_index, test_path in enumerate(config.test_paths):
            test_id = test_index + 1
            print(f'Epoch {epoch_num}: Evalutate test{test_id}')
            current_test_df = test_df[test_df['test_id'] == test_id]

            current, th_maps, th_cl_maps = calcmAP(model, test_df, config, path=test_path, confidence=confidence, thresholds=thresholds)
            r = [model_name, epoch_num, f'test{test_id}', current]
            for index, iou in enumerate(thresholds):
                print('mAP@%.2f: %.4f' % (iou, th_maps[index]))
                r.append(th_maps[index])
                ms = ""
                for cl in range(config.num_classes):
                    ms += 'AP@%.2f_class%d: %.4f; ' % (iou, cl, th_cl_maps[index, cl])
                    r.append(th_cl_maps[index, cl])
                print(ms)
                
            print('test%s mAP: %.4f' % (test_id, current))
            df.append(r)

    return pd.DataFrame(df, columns=['model_name', 'epoch', 'testset', 'mAP',	'mAP@0.50',	'AP@0.50_class0', 'AP@0.50_class1', 'AP@0.50_class2', 'mAP@0.70', 'AP@0.70_class0', 'AP@0.70_class1', 'AP@0.70_class2'])
