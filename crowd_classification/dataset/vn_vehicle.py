
from utils.config import Config
from glob import glob
import os
import shutil
from random import seed
from random import random
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def preprocessing(config: Config):
    raw_img_paths = glob(os.path.join(config.train_path, 'TrainImagePart*/'))
    output_path = os.path.join(config.train_path, 'train')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for img_path in raw_img_paths:
        annotation_paths = glob(os.path.join(img_path, 'PreprocessingVideo1*.txt'))
        classes_path = os.path.join(img_path, 'classes.txt')
        classes = []
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        # print(classes)
        for cl in classes:
            class_path = os.path.join(output_path, cl)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
        
        for anno_path in annotation_paths:
            first_line = ""
            with open(anno_path, 'r') as f:
                lines = f.readlines()
                first_line = lines[0].strip()
            label = int(first_line[0])
            class_path = os.path.join(output_path, classes[label])
            img_file = os.path.basename(anno_path)[:-4] + '.jpg'
            shutil.copyfile(os.path.join(img_path, img_file), os.path.join(class_path, img_file))

def load_data(config: Config):
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        width_shift_range=[-0.15, 0.15],
        height_shift_range=[-0.15, 0.15],
        horizontal_flip=True,
        rotation_range=20,
        brightness_range=[0.3,1.5],
        shear_range=15,
        zoom_range=[0.8,1.2],
        fill_mode='constant',
        cval=0
    )

    train_gen = datagen.flow_from_directory(config.train_path, class_mode='binary', batch_size=config.batch_size, target_size=(config.input_size, config.input_size))
    valid_gen = datagen.flow_from_directory(config.valid_path, class_mode='binary', batch_size=config.batch_size, target_size=(config.input_size, config.input_size))
    test_gen = datagen.flow_from_directory(config.test_path, class_mode='binary', batch_size=1, target_size=(config.input_size, config.input_size))

    return train_gen, valid_gen, test_gen


