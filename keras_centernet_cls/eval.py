from keras_centernet_cls.metrics import SaveBestScore, TestScore, calcScore
from keras_centernet_cls.losses import compile_model
from keras_centernet_cls.dataset.vn_vehicle import DataGenerator
from utils.config import Config
from models.centernet import create_model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import CSVLogger
from utils.loss_functions import CenterNetLosses
from glob import glob
import os
# from keras_centernet_cls.metrics import SaveBestmAP, TestmAP
import os
import numpy as np
import pandas as pd

#####TRAIN##########

def eval_models(valid_df, test_df, le, crowd_threshold, config: Config, model_prefix=None, model_ckpt_paths=[], model_garden={}):

    if model_prefix != None:
        model_ckpt_paths = glob(os.path.join(config.checkpoint_path, f'models/{model_prefix}*/'))
    print(model_ckpt_paths)
    count = 0
    result = None
    for ckpt_path in model_ckpt_paths:
        model_name = os.path.basename(ckpt_path)
        for k in model_garden:
            if k in model_name:
                model = model_garden[k]
                break
        versions = glob(os.path.join(ckpt_path, 'v*/'))
        for version in versions:
            version_name = os.path.basename(version)
            version_model_name = f'{model_name}_{version_name}'
            print(f'Evaluating model {version_model_name}')
            df = eval(model, os.path.join(version, 'every_epoch'), valid_df, test_df, le, crowd_threshold, config, model_name=version_model_name)
            if count == 0:
                result = df
            else:
                result = pd.concat([result, df])
            count += 1
    
    # if result != None:
    #     result.to_csv(os.path.join(config.checkpoint_path, f'eval_{model_prefix}.csv'), index=False, header=True)
    
        
    
def eval(model, checkpoint_path, valid_df, test_df, le, crowd_threshold, config: Config, confidence=0.25, generator=DataGenerator, model_name=None):

    model_names, epochs, testsets, accs, precs, recs, f1s = [], [], [], [], [], [], []

    ckpt_weights_files = glob(os.path.join(checkpoint_path, '*.hdf5'))

    for ckpt_file in ckpt_weights_files:
        model.load_weights(ckpt_file)
        ckpt_filename = os.path.basename(ckpt_file)
        epoch_num = int(ckpt_filename[: ckpt_filename.find('-')])
        
        accuracy, precision, recall, f1 = calcScore(model, valid_df, le, crowd_threshold, config, confidence=confidence)
        if model_name != None:
            model_names.append(model_name)
            print('%s valid: acc %.4f, prec %.4f, rec %.4f, f1 %.4f' % (model_name, accuracy, prediction, recall, f1))
        else:
            model_names.append(checkpoint_path)
            print('%s Valid: acc %.4f, prec %.4f, rec %.4f, f1 %.4f' % (model_name, accuracy, prediction, recall, f1))

        epochs.append(epoch_num)
        testsets.append('valid')
        accs.append(accuracy)
        precs.append(precision)
        recs.append(recall)
        f1s.append(f1)

        for test_index, test_path in enumerate(config.test_paths):
            test_id = test_index + 1
            print(f'Evalutate test{test_id}')
            current_test_df = test_df[test_df['test_id'] == test_id]

            accuracy, prediction, recall, f1 = calcScore(model, current_test_df, le, crowd_threshold, config, confidence=confidence, path=test_path)
            if model_name != None:
                model_names.append(model_name)
                print('%s test%d: acc %.4f, prec %.4f, rec %.4f, f1 %.4f' % (model_name, test_id, accuracy, prediction, recall, f1))
            else:
                model_names.append(checkpoint_path)
                print('test%d: acc %.4f, prec %.4f, rec %.4f, f1 %.4f' % (test_id, accuracy, prediction, recall, f1))

            epochs.append(epoch_num)
            testsets.append(f'test{test_id}')
            accs.append(accuracy)
            precs.append(precision)
            recs.append(recall)
            f1s.append(f1)

    return pd.DataFrame({
        'model_name': model_names,
        'epoch': epochs,
        'testset': testsets,
        'accuracy': accs,
        'precision': precs,
        'recall': recs,
        'f1': f1s,
    })
