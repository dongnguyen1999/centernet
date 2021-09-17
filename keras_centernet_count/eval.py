from keras_centernet_count.metrics import SaveBestScore, TestScore, calcMaeV0, calcScore
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
    result = pd.DataFrame([], columns=['model_name', 'epoch', 'testset', 'mae', 'mae_0.25', 'mae_0.25_0', 'mae_0.25_1', 'mae_0.25_2', 'mae_0.5', 'mae_0.5_0', 'mae_0.5_1', 'mae_0.5_2',  'mae_0.75', 'mae_0.75_0', 'mae_0.75_1', 'mae_0.75_2'])
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
    
        
    
def eval(model, checkpoint_path, valid_df, test_df, config: Config, confidences=[0.25, 0.5, 0.75], generator=DataGenerator, model_name=None):

    df = []

    ckpt_weights_files = glob(os.path.join(checkpoint_path, '*.hdf5'))
    if model_name == None:
        model_name = checkpoint_path

    for ckpt_file in ckpt_weights_files:
        model.load_weights(ckpt_file)
        ckpt_filename = os.path.basename(ckpt_file)
        epoch_num = int(ckpt_filename[: ckpt_filename.find('-')])
        
        print(f'Epoch {epoch_num}: Evalutate valid')
        conf_maes = calcMaeV0(model, valid_df, config, confidences=confidences)
        sum_maes = np.sum(conf_maes, axis=1)
        current = np.mean(sum_maes)
        r = [model_name, epoch_num, f'valid', current]
        print('average mae: %.4f' % current)

        for index, conf in enumerate(confidences):
            print('mae@%.2f: %.4f' % (conf, sum_maes[index]))
            r.append(sum_maes[index])

            for cls in range(config.num_classes):
                print('mae@%.2f_class_%d: %.4f' % (conf, cls, conf_maes[index, cls]))
                r.append(conf_maes[index, cls])
        
        df.append(r)

        for test_index, test_path in enumerate(config.test_paths):
            test_id = test_index + 1
            print(f'Epoch {epoch_num}: Evalutate test{test_id}')
            current_test_df = test_df[test_df['test_id'] == test_id]
            conf_maes = calcMaeV0(model, current_test_df, config, confidences=confidences, path=test_path)
            sum_maes = np.sum(conf_maes, axis=1)
            current = np.mean(sum_maes)
            r = [model_name, epoch_num, f'valid', current]
            print('average mae: %.4f' % current)

            for index, conf in enumerate(confidences):
                print('mae@%.2f: %.4f' % (conf, sum_maes[index]))
                r.append(sum_maes[index])

                for cls in range(config.num_classes):
                    print('mae@%.2f_class_%d: %.4f' % (conf, cls, conf_maes[index, cls]))
                    r.append(conf_maes[index, cls])
            df.append(r)


    return pd.DataFrame(df, columns=['model_name', 'epoch', 'testset', 'mae', 'mae_0.25', 'mae_0.25_0', 'mae_0.25_1', 'mae_0.25_2', 'mae_0.5', 'mae_0.5_0', 'mae_0.5_1', 'mae_0.5_2',  'mae_0.75', 'mae_0.75_0', 'mae_0.75_1', 'mae_0.75_2'])
