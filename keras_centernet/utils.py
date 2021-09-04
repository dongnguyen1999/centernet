from utils.config import Config
import numpy as np
import cv2

def normalize_image(image):
  """Normalize the image for the Hourglass network.
  # Arguments
    image: BGR uint8
  # Returns
    float32 image with the same shape as the input
  """
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

def heatmap(bbox, image_size, config: Config, sigma=2): # image size (h, w)
    def get_coords(box):
      x1, y1, x2, y2, label = box
      image_height, image_width = image_size
      w = x2 - x1
      h = y2 - y1

      xc = x1 + 0.5*w
      yc = y1 + 0.5*h
      
      x_c, y_c, width, height = (
        xc*config.output_size/image_width,
        yc*config.output_size/image_height,
        w*config.output_size/image_width, 
        h*config.output_size/image_height
      ) # Get xc, yc, w, h in output map
      return x_c, y_c, width, height, label
    
    def get_heatmap(p_x, p_y):
      # Ref: https://www.kaggle.com/diegojohnson/centernet-objects-as-points
      X1 = np.linspace(1, config.output_size, config.output_size)
      Y1 = np.linspace(1, config.output_size, config.output_size)
      [X, Y] = np.meshgrid(X1, Y1)
      X = X - floor(p_x)
      Y = Y - floor(p_y)
      D2 = X * X + Y * Y
      E2 = 2.0 * sigma ** 2
      Exponent = D2 / E2
      heatmap = np.exp(-Exponent)
      return heatmap

    coors = [] # list of [x, y]
    size = sigma+1
    y_ = size
    while y_ > -size - 1:
      x_ = -size
      while x_ < size + 1:
        coors.append([y_, x_])
        x_ += 1
      y_ -= 1

    output_layer = np.zeros((config.output_size, config.output_size,(2*config.num_classes + 4))) 

    for box in bbox:
      u, v, w, h, label = get_coords(box)
      if w == 0 or h == 0: continue
      # print(u, v, w, h)
      for coor in coors:
        try:
          output_layer[int(v)+coor[0], int(u)+coor[1], config.num_classes] = v%1
          output_layer[int(v)+coor[0], int(u)+coor[1], config.num_classes+1] = u%1
          output_layer[int(v)+coor[0], int(u)+coor[1], config.num_classes+2] = h
          output_layer[int(v)+coor[0], int(u)+coor[1], config.num_classes+3] = w
        except:
          pass
      heatmap = get_heatmap(u, v)
      print(label)
      output_layer[:,:,label] = np.maximum(output_layer[:,:,label], heatmap[:,:])
      output_layer[int(v), int(u), config.num_classes+4 + label] = 1


    
    # print(hm.shape, reg.shape, wh.shape)
  
    return output_layer
    