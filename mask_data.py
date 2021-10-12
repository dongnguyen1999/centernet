from PIL.Image import new
import numpy as np
from glob import glob 
import os
import cv2 
from matplotlib import pyplot as plt
import pandas as pd

from inference.model.decode import create_mask
from utils.tool import auto_concat_rfds

# DATA_PATH = r'C:\vehicle-data\video1_all\train'
# OUTPUT_PATH = r'C:\vehicle-data\mask_full\train'
# ANNOTATION_PATH = r"C:\vehicle-data\video1_all\train\_annotations_custom_v2.txt"

# DATA_PATH = r'C:\vehicle-data\video1_all\valid'
# OUTPUT_PATH = r'C:\vehicle-data\mask_full\valid'
# ANNOTATION_PATH = r"C:\vehicle-data\video1_all\valid\_annotations_custom_v2.txt"

# DATA_PATH = r'C:\vehicle-data\video1_all\test'
# OUTPUT_PATH = r'C:\vehicle-data\mask_full\test1'
# ANNOTATION_PATH = r"C:\vehicle-data\video1_all\test\_annotations_custom_v2.txt"

# DATA_PATH = r'C:\vehicle-data\video9\test'
# OUTPUT_PATH = r'C:\vehicle-data\mask_full\test2'
# ANNOTATION_PATH = r"C:\vehicle-data\video9\test\_annotations_custom_v2.txt"

# DATA_PATH = r'C:\vehicle-data\video12_16\test'
# OUTPUT_PATH = r'C:\vehicle-data\mask_full\test3'
# ANNOTATION_PATH = r"C:\vehicle-data\video12_16\test\_annotations_custom_v2.txt"


DATA_PATH = r'C:\vehicle-data\video12_16_xoay\test'
OUTPUT_PATH = r'C:\vehicle-data\mask_full\test4'
ANNOTATION_PATH = r"C:\vehicle-data\video12_16_xoay\test\_annotations_custom_v2.txt"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

mouseX,mouseY = 0,0
def get_mouse_click_coord(event, x, y, flags, param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX,mouseY = x,y

cv2.namedWindow('Data')
cv2.setMouseCallback('Data', get_mouse_click_coord)

first_frame = True

pts = []
mask = None
# mask = create_mask(pts)

# for img_path in glob(os.path.join(DATA_PATH, '*.jpg')):
#     filename = os.path.basename(img_path)
#     img = cv2.imread(img_path)
#     im_h, im_w = img.shape[:2]
#     if first_frame == True:
#         cv2.imshow("Data", img)
#         while True:
#             key = cv2.waitKey(0)
#             if key == ord('n'):
#                 pts.append([mouseX, mouseY])
#                 print('Cap coord: ', mouseX,mouseY)
#             elif key == ord('e'):
#                 print('Record mask: ', pts)
#                 pts = np.array(pts, dtype=np.int32)
#                 mask = create_mask(pts, (im_w, im_h))
#                 break
#         first_frame = False
    
#     print(f'Processing image {img_path}')
#     img[mask == False] = 0
#     cv2.imshow("Data", img)

#     cv2.imwrite(os.path.join(OUTPUT_PATH, filename), img)


# names=['filename', 'x1', 'y1', 'x2', 'y2', 'label']
# df = pd.read_csv(ANNOTATION_PATH, names=names)

# new_anno = []
# exclude_count = 0

# for img_id in df['filename'].unique():
#     filename = os.path.basename(img_id)
#     img = cv2.imread(os.path.join(OUTPUT_PATH, filename))
#     im_h, im_w = img.shape[:2]

#     boxes = df[df['filename'] == img_id][['x1', 'y1', 'x2', 'y2', 'label']].values

#     for box in boxes:
#         x1, y1, x2, y2, label = box
#         w, h = x2-x1, y2-y1

#         area = w*h

#         zeros = img[y1:y2, x1:x2, 0]
#         zero_area = zeros[np.where(zeros == 0)].size

#         if (area == 0) or ((zero_area / area) > 0.6):
#             exclude_count += 1
#             print(f'Exclude box ({x1},{x2},{y1},{y2}). Total excludes {exclude_count} items')
#         else:
#             new_anno.append([img_id, x1, y1, x2, y2, label])

# new_anno = pd.DataFrame(new_anno)
# new_anno.to_csv(os.path.join(OUTPUT_PATH, '_annotations.csv'), index=False, header=False)

auto_concat_rfds(r'C:\vehicle-data\mask_full\mask_safe_mosaic')





    

