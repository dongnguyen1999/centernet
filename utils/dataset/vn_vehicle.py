import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imgaug import augmenters
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence
from utils.config import Config
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from glob import glob
import math

def preprocessing(config: Config, limit=None):
    output_path = os.path.join(config.data_base, 'masked_train')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    names=['filename', 'x1', 'y1', 'x2', 'y2', 'label']
    train_df = pd.read_csv(os.path.join(config.train_path, config.annotation_filename), names=names)
    mask_df = pd.read_csv(os.path.join(config.train_path, 'mask.csv'), names=['x1', 'y1', 'x2', 'y2'])
    image_ids = train_df['filename'].unique()
    img_ids, x1s, y1s, x2s, y2s, labels = [], [], [], [], [], []
    for image_id in image_ids:
        filename = os.path.basename(image_id)
        # load source image
        img = cv2.imread(os.path.join(config.train_path, filename))
        im_h, im_w = img.shape[:2]

        mask = np.ones((im_h, im_w))
        mask_coords = mask_df[['x1', 'y1', 'x2', 'y2']].values
        for ignore_box in mask_coords:
            x1, y1, x2, y2 = ignore_box
            mask[y1:y2, x1:x2] = 0
        
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
        img = np.multiply(img, mask)
        cv2.imwrite(os.path.join(output_path, filename), img)

        boxes = train_df[train_df.filename == image_id]
        boxes = boxes[['x1', 'y1', 'x2', 'y2', 'label']].values
        for box in boxes:
            x1, y1, x2, y2, label = box
            is_error = False
            for ignore_box in mask_coords:
                ix1, iy1, ix2, iy2 = ignore_box
                dx = min(x2, ix2) - max(x1, ix1)
                dy = min(y2, iy2) - max(y1, iy1)
                if (dx > 0) and (dy > 0):
                    is_error = True
                    break
            if not is_error:
                img_ids.append(image_id)
                x1s.append(x1)
                y1s.append(y1)
                x2s.append(x2)
                y2s.append(y2)
                labels.append(label)
    
    assert len(img_ids) == len(x1s) and len(x1s) == len(y1s) and len(y1s) == len(x2s) and len(x2s) == len(y2s) and len(y2s) == len(labels)
    df = pd.DataFrame(data={
        'filename': img_ids, 'x1': x1s, 'y1': y1s, 'x2': x2s, 'y2': y2s, 'label': labels
    })
    df.to_csv(os.path.join(output_path, '_annotations.csv'), index=False, header=False)
    return df


def load_data(config: Config):    
    names=['filename', 'x1', 'y1', 'x2', 'y2', 'label']
    train_df = pd.read_csv(os.path.join(config.train_path, config.annotation_filename), names=names)
    test_df = pd.read_csv(os.path.join(config.test_path, config.annotation_filename), names=names)
    valid_df = pd.read_csv(os.path.join(config.valid_path, config.annotation_filename), names=names)
    le = LabelEncoder()
    train_df = train_df[train_df.label != 'person']
    train_df.label = le.fit_transform(train_df.label)
    test_df.label = le.transform(test_df.label)
    valid_df.label = le.transform(valid_df.label)

    print('Train data size: %d' % len(train_df.filename.unique()))
    print('Test data size: %d' % len(test_df.filename.unique()))
    print('Valid data size: %d' % len(valid_df.filename.unique()))

    return train_df, test_df, valid_df, le


def data_augmentation():
    sometimes = lambda x: augmenters.Sometimes(0.5, x)

    augment_sequential = augmenters.Sequential([
        sometimes(
            augmenters.OneOf([
                augmenters.Add((-10, 10), per_channel=0.5),
                augmenters.Multiply((0.9, 1.1), per_channel=0.5),
                augmenters.ContrastNormalization((0.9, 1.1), per_channel=0.5)
            ])
        ),
        augmenters.AdditiveGaussianNoise(scale=(0, 0.08*255)),
        #sometimes(augmenters.Rotate((-90, 90))),
        sometimes(augmenters.Fliplr(0.5)),
        #sometimes(augmenters.Crop(percent=(0, 0.2))),
        sometimes(augmenters.Flipud(0.5))
    ], random_order=False)

    return augment_sequential

class DataGenerator(Sequence):
    def __init__(self, dataframe, config: Config, mode='fit'):
        self.image_ids = dataframe[config.image_id].unique()
        self.df = dataframe
        self.train_path = config.train_path 
        self.valid_path = config.test_path if mode == 'test' else config.valid_path
        if self.valid_path == None:
            self.valid_path = config.train_path
        self.batch_size = config.batch_size
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.is_train = mode == 'fit'
        self.num_classes = config.num_classes
        self.seed = config.seed
        self.num_output_layers = self.num_classes + 4 # The first n layers is heatmap for each class, 2 next layers for h-offset and w-offset, 2 last layers for h-size and w-size
        if self.is_train:
            self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_ids)/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_ids[idx*self.batch_size : (idx+1)*self.batch_size]
        if self.is_train:
            return self.train_generator(batch_x)
        return self.valid_generate(batch_x)

    def on_epoch_end(self):
        if(self.is_train):
            self.image_ids = shuffle(self.image_ids)

    def _process_label(self, records, image_size): # image_size (w, h) is source image size
        boxes = records[['x1', 'y1', 'x2', 'y2', 'label']].values
        image_width, image_height = image_size
        output_height, output_width = self.output_size, self.output_size
        
        #PROCESS LABELS
        # This output for y-true, stacks some more layers for the exact center point coord for each class
        output_layer = np.zeros((output_height,output_width,(self.num_output_layers + self.num_classes))) 
        
        for box in boxes:
            x1, y1, x2, y2, category = box
            w = x2 - x1
            h = y2 - y1
            if w == 0 or h == 0: continue

            xc = x1 + 0.5*w
            yc = y1 + 0.5*h
            
            x_c, y_c, width, height = (
                xc*output_height/image_width,
                yc*output_height/image_height,
                w*output_height/image_width, 
                h*output_height/image_height
            ) # Get xc, yc, w, h in output map

            # print(x_c, y_c)
            heatmap = ((np.exp(-(((np.arange(output_width)-x_c)/(width/10))**2)/2)).reshape(1,-1) 
                    * (np.exp(-(((np.arange(output_height)-y_c)/(height/10))**2)/2)).reshape(-1,1))

            output_layer[:,:,category] = np.maximum(output_layer[:,:,category], heatmap[:,:]) #heatmap: R[0, 1] (h, w, c)

            # offset R[0, 1] (h, w, 2)
            output_layer[int(y_c//1),int(x_c//1), self.num_classes] = y_c%1 #height offset
            output_layer[int(y_c//1),int(x_c//1), self.num_classes + 1] = x_c%1 #width offset

            # size R[0, 1] (h, w, 2)
            output_layer[int(y_c//1),int(x_c//1), self.num_classes + 2]= height/output_height #scaled box height
            output_layer[int(y_c//1),int(x_c//1), self.num_classes + 3]= width/output_width #scaled box width

            # exact center point for compute in validation bool (h, w, c)
            output_layer[int(y_c//1),int(x_c//1), self.num_classes+4 + category] = 1 #center point

        return output_layer

    def train_generator(self, batch_x):
        batch_imgs = []
        batch_segs = []
        for filename in batch_x:
            records = self.df[self.df['filename'] == filename]
            filename = os.path.basename(filename)

            # load source image
            img = cv2.imread(f'{self.train_path}/{filename}')
            im_h, im_w = img.shape[:2]
            img = cv2.resize(img, (self.input_size, self.input_size))

            output_layer = self._process_label(records, (im_w, im_h))

            #images_aug, segmaps_aug = seq(images=[img], heatmaps=[output_layer.astype(np.float32)])

            #batch_imgs.append(images_aug[0])
            #batch_segs.append(segmaps_aug[0])

            batch_imgs.append(img)
            batch_segs.append(output_layer)

        batch_imgs = np.array(batch_imgs, np.float32) /255 #normalize image
        batch_segs = np.array(batch_segs, np.float32)

        return batch_imgs, batch_segs #X, y


    def valid_generate(self, batch_x):
        batch_imgs = []
        batch_segs = []
        
        for filename in batch_x:
            records = self.df[self.df['filename'] == filename]
            filename = os.path.basename(filename)

            # load source image
            img = cv2.imread(f'{self.valid_path}/{filename}')
            im_h, im_w = img.shape[:2]
            img = cv2.resize(img, (self.input_size, self.input_size))

            output_layer = self._process_label(records, (im_w, im_h))

            #images_aug, segmaps_aug = seq(images=[img], heatmaps=[output_layer.astype(np.float32)])


            batch_imgs.append(img)
            batch_segs.append(output_layer)


        batch_imgs = np.array(batch_imgs, np.float32) /255
        batch_segs = np.array(batch_segs, np.float32)

        return batch_imgs, batch_segs #X, y


def test_dataset(train_df, config: Config, mode='fit'):
    mygen = DataGenerator(train_df, config, mode=mode)

    for count, (x,y) in enumerate(mygen):
        # print(x.shape)
        # print(y.shape)
        x = x[0]
        y = y[0]
        # print(y.shape)
        for category in range(config.num_classes):
            points = np.argwhere(y[:,:, config.num_classes+4 + category] == 1)

            for y1,x1 in points:
                # print(y1,x1)
                offsety = y[:,:, config.num_classes + 0][y1,x1]
                offetx = y[:,:, config.num_classes + 1][y1,x1]
                h = y[:,:, config.num_classes + 2][y1,x1] * config.input_size/4
                w = y[:,:, config.num_classes + 3][y1,x1] * config.input_size/4

                x1, y1 = x1+offetx, y1+offsety 

                xmin = int((x1-w/2)*4)
                xmax = int((x1+w/2)*4)
                ymin = int((y1-h/2)*4)
                ymax = int((y1+h/2)*4)

                cv2.rectangle(x, (xmin, ymin), (xmax, ymax), (0,255,255), 2)
                cv2.circle(x, (int(x1*4),int(y1*4)), 2, (255,0,0), -1) 

        #cv2.imshow('djpg',y[:,:,1]*255)
        #cv2.imshow('drawjpg',x)
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        ax.set_axis_off()
        ax.imshow(x)
        return x, y
