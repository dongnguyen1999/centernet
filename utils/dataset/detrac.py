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
import xml.etree.ElementTree as ET
import math

def preprocessing(config: Config, limit=None):
    image_path = os.path.join(config.train_path, 'image')
    xml_path = os.path.join(config.train_path, 'xml')
    img_ids, x1s, y1s, x2s, y2s, labels = [], [], [], [], [], []
    subfolders = glob(os.path.join(image_path, '*'))

    mask_path = os.path.join(config.train_path, 'mask')
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    if limit != None:
        subfolders = subfolders[:limit]
    for path in subfolders:
        folder_name = os.path.basename(path)
        xml_name = f'{folder_name}_v3.xml'
        folder_path = os.path.join(image_path, folder_name)
        tree = ET.parse(os.path.join(xml_path, xml_name))
        img = cv2.imread(os.path.join(folder_path, 'img00001.jpg'))
        im_h, im_w = img.shape[:2]

        root = tree.getroot()
        mask = np.ones((im_h, im_w))
        
        for ignore_box in root.find('ignored_region').findall('box'):
            x, y, w, h = float(ignore_box.attrib['left']), float(ignore_box.attrib['top']), float(ignore_box.attrib['width']), float(ignore_box.attrib['height'])
            x, y, w, h = [math.floor(i) for i in (x, y, w, h)]
            mask[y:y+h, x:x+w] = 0

        np.save(os.path.join(mask_path, f'{folder_name}_mask'), mask)

        for frame in root.findall('frame'):
            frame_num = int(frame.attrib['num'])
            img_name = "img%05d.jpg" % frame_num
            img_name = '$'.join([folder_name, img_name])
            for target in frame.find('target_list').findall('target'):
                box = target.find('box')
                x, y, w, h = float(box.attrib['left']), float(box.attrib['top']), float(box.attrib['width']), float(box.attrib['height'])
                x1, y1, x2, y2 = x, y, x+w, y+h
                label = target.find('attribute').attrib['vehicle_type']
                img_ids.append(img_name)
                x1s.append(x1)
                y1s.append(y1)
                x2s.append(x2)
                y2s.append(y2)
                labels.append(label)

    assert len(img_ids) == len(x1s) and len(x1s) == len(y1s) and len(y1s) == len(x2s) and len(x2s) == len(y2s) and len(y2s) == len(labels)
    df = pd.DataFrame(data={
        'filename': img_ids, 'x1': x1s, 'y1': y1s, 'x2': x2s, 'y2': y2s, 'label': labels
    })
    df.to_csv(os.path.join(config.train_path, '_annotations.csv'), index=False, header=False)
    return df


def load_data(config: Config):    
    names=['filename', 'x1', 'y1', 'x2', 'y2', 'label']
    df = pd.read_csv(os.path.join(config.train_path, config.annotation_filename), names=names)

    # print(len(df.label.unique()))
    # 13 classes
    # ['Suv' 'Sedan' 'Taxi' 'Van' 'Truck-Box-Large' 'Hatchback' 'Bus' 'Police'
    # 'MiniVan' 'Truck-Box-Med' 'Truck-Util' 'Truck-Pickup' 'Truck-Flatbed']

    le = LabelEncoder()
    df.label = le.fit_transform(df.label)

    image_ids = df[config.image_id].unique()
    print('Image num: %d' % len(image_ids))

    train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=config.seed)

    train_ids, valid_ids = train_test_split(train_ids, test_size=0.2, random_state=config.seed)

    print('Train num: %d' % len(train_ids))
    print('Valid num: %d' % len(valid_ids))
    print('Test num: %d' % len(test_ids))

    train_df = df[df[config.image_id].isin(train_ids)]    
    valid_df = df[df[config.image_id].isin(valid_ids)]    
    test_df = df[df[config.image_id].isin(test_ids)]    

    return train_df, test_df, valid_df, le

class DataGenerator(Sequence):
    def __init__(self, dataframe, config: Config, mode='fit'):
        self.image_ids = dataframe[config.image_id].unique()
        self.df = dataframe
        self.train_path = os.path.join(config.train_path, 'image')
        self.valid_path = config.test_path if mode == 'test' else config.valid_path
        self.mask_path = os.path.join(config.train_path, 'mask')
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
        output_layer = np.zeros((output_height, output_width, (self.num_output_layers + self.num_classes))) 
        print(self.num_output_layers + self.num_classes)
        
        for box in boxes:
            x1, y1, x2, y2, category = box
            category = int(category)
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
            subfolder, filename = filename.split('$')
            filename = os.path.join(subfolder, filename)

            # load source image
            img = cv2.imread(os.path.join(self.train_path, filename))

            with open(os.path.join(self.mask_path, f'{subfolder}_mask.npy'), 'rb') as f:
                mask = np.load(f).astype(np.float)
                print(mask.shape)
                mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
                img = np.multiply(img, mask)

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
            subfolder, filename = filename.split('$')
            filename = os.path.join(subfolder, filename)

            # load source image
            img = cv2.imread(os.path.join(self.valid_path, filename))

            with open(os.path.join(self.mask_path, f'{subfolder}_mask.npy'), 'rb') as f:
                mask = np.load(f)
                img = np.multiply(img, mask)

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
