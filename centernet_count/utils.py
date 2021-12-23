from utils.config import Config
import numpy as np
import cv2

def normalize_image(image):
  mean = [0.40789655, 0.44719303, 0.47026116]
  std = [0.2886383, 0.27408165, 0.27809834]
  return ((np.float32(image) / 255.) - mean) / std

from math import floor

def get_boxes(bbox):
  boxes = []
  for box in bbox:
    box = box[1:-1].split(',')
    box = [float(b) for b in box]
    box = [int(b) for b in box]
    boxes.append(box)

  boxes = np.array(boxes, dtype=np.int32)
  return boxes

def heatmap(bbox, image_size, config: Config): # image size (h, w)
    sigma = (config.output_size // 128) * 5
    size = (config.output_size // 128) * 10

    def get_coords(box):
      x1, y1, x2, y2, label = box
      image_height, image_width = image_size
      w = x2 - x1
      h = y2 - y1

      xc = x1 + 0.5*w
      yc = y1 + 0.5*h
      
      x_c, y_c, x, y, width, height = (
        xc*config.input_size/image_width,
        yc*config.input_size/image_height,
        xc*config.output_size/image_width,
        yc*config.output_size/image_height,
        w*config.output_size/image_width, 
        h*config.output_size/image_height
      ) # Get xc, yc, w, h in output map
      return x_c, y_c, x, y, width, height, int(label)
    
    def get_heatmap(p_x, p_y):
      # Ref: https://www.kaggle.com/diegojohnson/centernet-objects-as-points
      X1 = np.linspace(1, config.input_size, config.input_size)
      Y1 = np.linspace(1, config.input_size, config.input_size)
      [X, Y] = np.meshgrid(X1, Y1)
      X = X - floor(p_x)
      Y = Y - floor(p_y)
      D2 = X * X + Y * Y
      E2 = 2.0 * sigma ** 2
      Exponent = D2 / E2
      heatmap = np.exp(-Exponent)
      return heatmap

    coors = [] # list of [x, y]    
    y_ = size
    while y_ > -size - 1:
      x_ = -size
      while x_ < size + 1:
        coors.append([y_, x_])
        x_ += 1
      y_ -= 1

    hm_output = np.zeros((config.input_size, config.input_size, config.num_classes)) 
    reg_output = np.zeros((config.input_size, config.input_size, 2))
    wh_output = np.zeros((config.input_size, config.input_size, 2))

    for box in bbox:
      u, v, x, y, w, h, label = get_coords(box) # u, v is (x, y) in input coord; x,y,w,h is x,y,width,height in strides 1/4 coord
      if w <= 0 or h <= 0: continue
      # print(u, v, w, h)
      for coor in coors:
        try:
          reg_output[int(v)+coor[0], int(u)+coor[1], 0] = y%1
          reg_output[int(v)+coor[0], int(u)+coor[1], 1] = x%1

          wh_output[int(v)+coor[0], int(u)+coor[1], 0] = h
          wh_output[int(v)+coor[0], int(u)+coor[1], 1] = w
        except:
          pass
      heatmap = get_heatmap(u, v)
      hm_output[:,:,label] = np.maximum(hm_output[:,:,label], heatmap[:,:])

    hm = np.zeros((config.output_size, config.output_size, (2*config.num_classes + 1))) 
    reg = np.zeros((config.output_size, config.output_size, 3)) 
    wh = np.zeros((config.output_size, config.output_size, 3)) 

    for i in range(config.num_classes):
      hm[:,:,i] = cv2.resize(hm_output[:,:,i], (config.output_size, config.output_size))
    
    for i in range(2):
      reg[:,:,i] = cv2.resize(reg_output[:,:,i], (config.output_size, config.output_size))
      wh[:,:,i] = cv2.resize(wh_output[:,:,i], (config.output_size, config.output_size))
    
    # Compute masks
    for box in bbox:
      u, v, x, y, w, h, label = get_coords(box) # u, v is (x, y) in input coord; x,y,w,h is x,y,width,height in strides 1/4 coord
      if w <= 0 or h <= 0: continue
      hm[int(y), int(x), config.num_classes + label] = 1
      hm[int(y), int(x), 2*config.num_classes] = 1
      reg[int(y), int(x), 2] = 1
      wh[int(y), int(x), 2] = 1
    
    # print(hm.shape, reg.shape, wh.shape)
  
    return hm, reg, wh
    