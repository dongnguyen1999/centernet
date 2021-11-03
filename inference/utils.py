from inference.config import Config
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def create_mask(points, size=(512, 512)):
  img = np.zeros((size[1], size[0], 3), dtype=np.float32)
  points = points.reshape((-1, 1, 2))
  mask = cv2.fillPoly(img, pts = [points], color=(1,1,1))
  mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
  mask = mask.astype(np.bool)
  return mask

def get_masked_img(img, mask):
  img[mask == False] = 0
  return img

def compute_count_score(counts, weights=[1, 2, 3]):
  return counts[0]*weights[0] + counts[1]*weights[1] + counts[2]*weights[2]

def visualize(img, end_label, model_result, cls_time, bs_time, dt_time, pred_time, display=False):
  boxes = []
  scores = []

  color_scheme = [(0,255,255), (255,0,0), (0,0,255), (0,127,127), (127,255,127), (255,255,0)]
  end_label_map = ['Low', 'Medium', 'High', 'Normally', 'Slowly', 'Traffic jam']

  cls_fps = 1000/(cls_time + 1e-6)
  bs_fps = 1000/(bs_time + 1e-6)
  dt_fps = 1000/(dt_time + 1e-6)
  fps = 1000/(pred_time + 1e-6)

  im_h, im_w = img.shape[:2]

  result = {}

  # print(cls_time, dt_time, pred_time)
  if end_label < 3:
    box_and_score = model_result

    label_map = ['2-wheel', '4-wheel', 'priority']
    
    count = np.array([0, 0, 0])

    nb_cols = box_and_score.shape[1]

    if nb_cols > 2: 
      # model_result is fully bounding box result
      # visualize boxes
      number_of_rect = len(box_and_score)
      for i in range(number_of_rect):
        left, top, right, bottom, score, predicted_class = box_and_score[i, :]
        top = np.floor(top).astype('int32')
        left = np.floor(left).astype('int32')
        bottom = np.floor(bottom).astype('int32')
        right = np.floor(right).astype('int32')
        predicted_class = int(predicted_class)
        label = '%s %.2f' % (label_map[predicted_class], score)

        count[predicted_class] += 1
        
        #print(top, left, right, bottom)
        cv2.rectangle(img, (left, top), (right, bottom), color_scheme[predicted_class], 1)
        cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX ,  
                    0.5, color_scheme[predicted_class], 1, cv2.LINE_AA) 
        boxes.append([left, top, right-left, bottom-top])
        scores.append(score)

    # cv2.rectangle(img, (1, 1), (output_size[0]-1, 50), (0, 0, 0), -1)

    # for i in range(3):
    #   cv2.putText(img, f'{label_map[i]}: {count[i]}', ((i* 110) + 12, 23), cv2.FONT_HERSHEY_SIMPLEX ,  
    #               0.5, color_scheme[i], 1, cv2.LINE_AA)

    # cv2.putText(img, f'Total: {np.sum(count)}', (12, 42), cv2.FONT_HERSHEY_SIMPLEX ,  
    #               0.5, (255,255,255), 1, cv2.LINE_AA) 

    # cv2.putText(img, 'FPS: %.2f (cls: %.2f, dt: %.2f)' % (fps, cls_fps, dt_fps), (100, 42), cv2.FONT_HERSHEY_SIMPLEX ,  
    #               0.5, (255,255,255), 1, cv2.LINE_AA)

    # cv2.putText(img, f'Count score: {compute_count_score(count)}', (output_size[0]-142, 23), cv2.FONT_HERSHEY_SIMPLEX ,  
    #               0.5, (255,255,255), 1, cv2.LINE_AA) 

    # cv2.putText(img, f'Label: {end_label_map[end_label]}', (output_size[0]-195, 42), cv2.FONT_HERSHEY_SIMPLEX ,  
    #               0.5, (255,255,255), 1, cv2.LINE_AA) 

    cv2.rectangle(img, (1, 1), (im_w-1, im_h-1), (0, 255, 0), 3)

    result = {
      'label': end_label,
      'label_name': end_label_map[end_label],
      'total_count': np.sum(count),
      'count_score': compute_count_score(count),
      'count': count,
      'count_label_colors': [color_scheme[i] for i in range(len(count))],
      'fps': fps,
      'cls_fps': cls_fps,
      'dt_fps': dt_fps
    }
    

  else:
    diff_rate = result

    # cv2.rectangle(img, (1, 1), (output_size[0]-1, 50), (0, 0, 0), -1)

    # cv2.putText(img, 'FPS: %.2f (cls: %.2f, bs: %.2f)' % (fps, cls_fps, bs_fps), (12, 30), cv2.FONT_HERSHEY_SIMPLEX ,  
    #               0.5, (255,255,255), 1, cv2.LINE_AA) 

    # cv2.putText(img, 'Moving rate: %.2f%%' % diff_rate, (output_size[0]-166, 23), cv2.FONT_HERSHEY_SIMPLEX ,  
    #               0.5, (255,255,255), 1, cv2.LINE_AA) 

    # cv2.putText(img, f'Label: {end_label_map[end_label]}', (output_size[0]-219, 42), cv2.FONT_HERSHEY_SIMPLEX ,  
    #               0.5, (255,255,255), 1, cv2.LINE_AA)

    cv2.rectangle(img, (1, 1), (im_w-1, im_h-1), (0, 0, 255), 3)

    result = {
      'label': end_label,
      'label_name': end_label_map[end_label],
      'diff_rate': diff_rate,
      'fps': fps,
      'cls_fps': cls_fps,
      'bs_fps': bs_fps
    }


  if display == True:
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_axis_off()
    ax.imshow(img)

  return img, result