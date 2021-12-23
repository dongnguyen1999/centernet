from crowd_classification.metrics import SaveBestScore, TestScore, calcScore
from utils.config import Config
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import CSVLogger
from glob import glob
import os
import os
import numpy as np
import pandas as pd

def eval_models(valid_gen, test_gen, config: Config, model_prefix=None, model_ckpt_paths=[], model_garden={}, confidences=[0.25, 0.5, 0.75], eval_category='every_epoch'):

    if model_prefix != None:
        model_ckpt_paths = glob(os.path.join(config.logging_base, f'models/{model_prefix}*/'))
    # print(model_ckpt_paths)
    result = pd.DataFrame([], columns=['model_name', 'epoch', 'testset', 'conf_threshold', 'accuracy', 'precision', 'recall', 'f1', 'time'])
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
            for confidence in confidences:
                df = eval(model, os.path.join(version, eval_category), valid_gen, test_gen, config, confidence = confidence, model_name=version_model_name)
                result = pd.concat([result, df])
    
    result.to_csv(os.path.join(config.logging_base, f'eval_{model_prefix}.csv'), index=False, header=True)
    
        
    
def eval(model, checkpoint_path, valid_gen, test_gen, config: Config, confidence=0.25, model_name=None):

    model_names, epochs, testsets, conf_thresholds, accs, precs, recs, f1s, times = [], [], [], [], [], [], [], [], []

    ckpt_weights_files = glob(os.path.join(checkpoint_path, '*.hdf5'))

    for ckpt_file in ckpt_weights_files:
        model.load_weights(ckpt_file)
        ckpt_filename = os.path.basename(ckpt_file)
        epoch_num = int(ckpt_filename[: ckpt_filename.find('-')])
        
        print(f'Epoch {epoch_num}: Evalutate valid')
        accuracy, precision, recall, f1, time = calcScore(model, valid_gen, config, confidence=confidence)
        if model_name != None:
            model_names.append(model_name)
            print('%s valid: acc %.4f, prec %.4f, rec %.4f, f1 %.4f, runtime %.4fms' % (model_name, accuracy, precision, recall, f1, time))
        else:
            model_names.append(checkpoint_path)
            print('%s Valid: acc %.4f, prec %.4f, rec %.4f, f1 %.4f, runtime %.4fms' % (model_name, accuracy, precision, recall, f1, time))

        epochs.append(epoch_num)
        testsets.append('valid')
        conf_thresholds.append(confidence)
        accs.append(accuracy)
        precs.append(precision)
        recs.append(recall)
        f1s.append(f1)
        times.append(time)

        print(f'Epoch {epoch_num}: Evalutate test')
        accuracy, precision, recall, f1, time = calcScore(model, test_gen, config, confidence=confidence)
        if model_name != None:
            model_names.append(model_name)
            print('%s test: acc %.4f, prec %.4f, rec %.4f, f1 %.4f, runtime %.4fms' % (model_name, accuracy, precision, recall, f1, time))
        else:
            model_names.append(checkpoint_path)
            print('%s Test: acc %.4f, prec %.4f, rec %.4f, f1 %.4f, runtime %.4fms' % (model_name, accuracy, precision, recall, f1, time))

        epochs.append(epoch_num)
        testsets.append('test')
        conf_thresholds.append(confidence)
        accs.append(accuracy)
        precs.append(precision)
        recs.append(recall)
        f1s.append(f1)
        times.append(time)

    return pd.DataFrame({
        'model_name': model_names,
        'epoch': epochs,
        'testset': testsets,
        'conf_threshold': conf_thresholds,
        'accuracy': accs,
        'precision': precs,
        'recall': recs,
        'f1': f1s,
        'time': times,
    })
