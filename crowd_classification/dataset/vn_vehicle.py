
from utils.config import Config
from glob import glob
import os
import shutil
from random import seed
from random import random
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def load_directory(path):
    if not os.path.exists(path):
        raise ValueError("Input path is not exist!")

    df = []
    for subdir in glob(os.path.join(path, '*/')):
        label_name = os.path.basename(subdir[:-1])
        for img_file in glob(os.path.join(subdir, '*.jpg')):
            filename = os.path.basename(img_file)
            df.append([filename, label_name])
    
    return pd.DataFrame(df, columns=['filename', 'label'])

def generate_label(splited_path, label_df):
    for img_path in glob(os.path.join(splited_path, '*.jpg')):
        filename = os.path.basename(img_path)
        label = label_df[label_df['filename'] == filename][['label']].values[0,0]
        subdir = os.path.join(splited_path, label)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        shutil.move(img_path, os.path.join(subdir, filename))
        

def preprocessing(source_path, labeled_path):
    train_path = os.path.join(source_path, 'train')
    valid_path = os.path.join(source_path, 'valid')
    test_path = os.path.join(source_path, 'test')

    df = load_directory(labeled_path)

    generate_label(train_path, df)
    generate_label(valid_path, df)
    generate_label(test_path, df)



def load_data(config: Config):
    # datagen = ImageDataGenerator(
    #     rescale=1.0/255.0,
    #     width_shift_range=[-0.15, 0.15],
    #     height_shift_range=[-0.15, 0.15],
    #     rotation_range=20,
    #     brightness_range=[0.3,1.5],
    #     shear_range=15,
    #     zoom_range=[0.8,1.2],
    #     horizontal_flip=True,
    #     fill_mode='constant',
    #     cval=0
    # )

    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        width_shift_range=[-0.1, 0.1],
        height_shift_range=[-0.1, 0.1],
        rotation_range=20,
        # brightness_range=[0.3,1.5],
        shear_range=15,
        zoom_range=[0.8,1.2],
        # horizontal_flip=True,
        fill_mode='constant',
        cval=0
    )

    valid_test_gen = datagen = ImageDataGenerator(rescale=1.0/255.0)


    train_gen = datagen.flow_from_directory(config.train_path, class_mode='binary', batch_size=config.batch_size, target_size=(config.input_size, config.input_size))
    valid_gen = valid_test_gen.flow_from_directory(config.valid_path, class_mode='binary', batch_size=config.batch_size, target_size=(config.input_size, config.input_size))
    test_gen = valid_test_gen.flow_from_directory(config.test_path, class_mode='binary', batch_size=1, target_size=(config.input_size, config.input_size))

    return train_gen, valid_gen, test_gen


