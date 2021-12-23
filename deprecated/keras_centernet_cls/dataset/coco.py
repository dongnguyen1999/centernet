from utils.config import Config
from pycocotools.coco import COCO
import requests
import os
import pandas as pd

def downloadDataset(config: Config):
    # instantiate COCO specifying the annotations json path
    train_anno_file = os.path.join('coco', 'instances_train2017.json')
    coco = COCO(os.path.join(config.data_base, train_anno_file))

    # display COCO categories
    # cats = coco.loadCats(coco.getCatIds())
    # df = pd.DataFrame(cats)
    # print(df[df['supercategory'] == 'vehicle'])

    vehicle = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']

    # Specify a list of category names of interest
    catIds = coco.getCatIds(catNms=vehicle)
    # Get the corresponding image ids and images using loadImgs
    annoIds = coco.getAnnIds(catIds=catIds, iscrowd=False)
    imgIds = coco.getImgIds(catIds=catIds)
    # images = coco.loadImgs(imgIds[:1])
    print(imgIds)