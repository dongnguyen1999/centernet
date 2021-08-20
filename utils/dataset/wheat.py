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

def expand_bbox(bbox):
    box = bbox[1:-1].split(',')
    return [float(x) for x in box]

def load_data(config: Config):    
    train_df = pd.read_csv(os.path.join(config.data_base, config.annotation_filename))

    bbox = train_df.bbox.apply(lambda bbox_row: expand_bbox(bbox_row))

    train_df[['x', 'y', 'w', 'h']] = np.stack(bbox)

    train_df = train_df.drop('bbox', axis=1)

    image_ids = train_df.image_id.unique()
    print('Image num: %d' % len(image_ids))

    train_ids, valid_ids = train_test_split(image_ids, test_size=0.2)

    print('Train num: %d' % len(train_ids))
    print('Valid num: %d' % len(valid_ids))

    valid_df = train_df[train_df.image_id.isin(valid_ids)]
    train_df = train_df[train_df.image_id.isin(train_ids)]    

    return train_df, valid_df


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
        self.id_filed = config.image_id
        self.df = dataframe
        self.image_dir = config.train_path if mode == 'fit' else config.test_path
        self.batch_size = config.batch_size
        self.input_size = config.input_size
        self.output_size = self.input_size // 4 # Center output size with stride 4 
        self.is_train = mode == 'fit'
        self.num_classes = config.num_classes
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
            self.image_ids= shuffle(self.image_ids)


    def train_generator(self, batch_x):
        batch_imgs = []
        batch_segs = []
        output_height, output_width = self.output_size, self.output_size
        for image_id in batch_x:
            records = self.df[self.df[self.id_filed] == image_id]

            img = cv2.imread(f'{self.image_dir}/{image_id}.jpg')
            im_h, im_w = img.shape[:2]
            img = cv2.resize(img, (self.input_size, self.input_size))

            boxes = records[['x', 'y', 'w', 'h']].values

            #PROCESS LABELS
            output_layer=np.zeros((output_height,output_width,(self.num_output_layers + self.num_classes)))

            for box in boxes:
                x, y, w, h = box
                xc = x + 0.5*w
                yc = y + 0.5*h
                x_c, y_c, width, height = xc*output_height/im_w, yc*output_height/im_h, w*output_height/im_w, h*output_height/im_h # Get xc, yc, w, h in output map
                # print(x_c, y_c)

                category = 0 #not classify, just detect
                heatmap=((np.exp(-(((np.arange(output_width)-x_c)/(width/10))**2)/2)).reshape(1,-1) 
                        * (np.exp(-(((np.arange(output_height)-y_c)/(height/10))**2)/2)).reshape(-1,1))

                output_layer[:,:,category]=np.maximum(output_layer[:,:,category],heatmap[:,:]) #heatmap
                output_layer[int(y_c//1),int(x_c//1),1*self.num_classes+category]=1 #center point
                output_layer[int(y_c//1),int(x_c//1),2*self.num_classes]=y_c%1 #height offset
                output_layer[int(y_c//1),int(x_c//1),2*self.num_classes+1]=x_c%1 #width offset
                output_layer[int(y_c//1),int(x_c//1),2*self.num_classes+2]=height/output_height #scaled box height
                output_layer[int(y_c//1),int(x_c//1),2*self.num_classes+3]=width/output_width #scaled box width


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
        output_height,output_width=self.output_size,self.output_size
        for image_id in batch_x:
            records = self.df[self.df[self.id_filed] == image_id]

            img = cv2.imread(f'{self.image_dir}/{image_id}.jpg')
            im_h, im_w = img.shape[:2]
            img = cv2.resize(img, (self.input_size, self.input_size))

            boxes = records[['x', 'y', 'w', 'h']].values

            #PROCESS LABELS
            output_layer=np.zeros((output_height,output_width,(self.num_output_layers + self.num_classes)))

            for box in boxes:
                x, y, w, h = box
                xc = x + 0.5*w
                yc = y + 0.5*h
                x_c, y_c, width, height = xc*output_height/im_w, yc*output_height/im_h, w*output_height/im_w, h*output_height/im_h
                # print(x_c, y_c)

                category=0 #not classify, just detect
                heatmap=((np.exp(-(((np.arange(output_width)-x_c)/(width/10))**2)/2)).reshape(1,-1)
                                    *(np.exp(-(((np.arange(output_height)-y_c)/(height/10))**2)/2)).reshape(-1,1))

                output_layer[:,:,category]=np.maximum(output_layer[:,:,category],heatmap[:,:]) #heatmap
                output_layer[int(y_c//1),int(x_c//1),1*self.num_classes+category]=1 #center point
                output_layer[int(y_c//1),int(x_c//1),2*self.num_classes]=y_c%1 #height offset
                output_layer[int(y_c//1),int(x_c//1),2*self.num_classes+1]=x_c%1  #width offset
                output_layer[int(y_c//1),int(x_c//1),2*self.num_classes+2]=height/output_height #scaled box height
                output_layer[int(y_c//1),int(x_c//1),2*self.num_classes+3]=width/output_width #scaled box width

            #images_aug, segmaps_aug = seq(images=[img], heatmaps=[output_layer.astype(np.float32)])


            batch_imgs.append(img)
            batch_segs.append(output_layer)


        batch_imgs = np.array(batch_imgs, np.float32) /255
        batch_segs = np.array(batch_segs, np.float32)

        return batch_imgs, batch_segs #X, y


def test_dataset(train_df, config: Config):
    mygen = DataGenerator(train_df, config)

    for count, (x,y) in enumerate(mygen):
        # print(x.shape)
        # print(y.shape)
        x = x[0]
        y= y[0]

        points = np.argwhere(y[:,:,1] ==1)
        for y1,x1 in points:
            # print(x1,y1)
            offsety = y[:,:,2][y1,x1]
            offetx = y[:,:,3][y1,x1]
            
            h = y[:,:,4][y1,x1]*104
            w = y[:,:,5][y1,x1]*104

            x1, y1 = x1 + offetx, y1+offsety 

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
