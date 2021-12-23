from inference.config import Config
import numpy as np
import cv2
import matplotlib.pyplot as plt
import threading

def AsynTask(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


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
    mask = cv2.fillPoly(img, pts=[points], color=(1, 1, 1))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.bool)
    return mask


def get_masked_img(img, mask):
    img[mask == False] = 0
    return img


def compute_count_score(counts, weights=[1, 2, 2.5]):
    return counts[0] * weights[0] + counts[1] * weights[1] + counts[2] * weights[2]

def bgr_to_hex(bgr):
    colortuple = (bgr[2], bgr[1], bgr[0])
    return '#' + ''.join(f'{i:02X}' for i in colortuple)

def visualize(img, model_result, cls_time, bs_time, dt_time, pred_time, display=False, config=None):
    boxes = []
    scores = []

    color_scheme = [(255, 0, 0), (0, 255, 255), (0, 0, 255), (0, 127, 127), (127, 255, 127), (255, 255, 0)]
    end_label_map = ['Sparse Density', 'Low Density', 'Medium Density', 'Moving Normally', 'Moving Slowly', 'Traffic Jam']

    cls_fps = 1000 / (cls_time + 1e-6)
    bs_fps = 1000 / (bs_time + 1e-6)
    dt_fps = 1000 / (dt_time + 1e-6)
    fps = 1000 / (pred_time + 1e-6)

    result = {}

    # print(cls_time, dt_time, pred_time)
    end_label = model_result['label']
    aspect_ratio = config['aspect_ratio']

    if aspect_ratio <= (960 / 640):
        img = cv2.resize(img, (int(640 * aspect_ratio), 640))
    else:
        img = cv2.resize(img, (960, int(960 / aspect_ratio)))

    src_im_h, src_im_w = img.shape[:2]
    if config['show_diff_mask'] or config['show_foreground_mask']:
        sub_im_w = int(src_im_w / 2)

        if aspect_ratio <= (sub_im_w / src_im_h):
            img = cv2.resize(img, (int(src_im_h * aspect_ratio), src_im_h))
        else:
            img = cv2.resize(img, (sub_im_w, int(sub_im_w / aspect_ratio)))

        left_img = img
        right_img = img.copy()

    else:
        left_img = img

    im_h, im_w = left_img.shape[:2]

    if config['show_foreground_mask']:
        foreground_mask = model_result['foreground_mask']
        foreground_mask = cv2.resize(foreground_mask, (im_w, im_h))
        right_img[foreground_mask > 0] = (0, 255, 255)

    if config['show_diff_mask']:
        diff_mask = model_result['diff_mask']
        diff_mask = cv2.resize(diff_mask, (im_w, im_h))
        right_img[diff_mask > 0] = (255, 0, 0)

    if end_label < 3:
        box_and_score = model_result['detections']

        label_map = ['2-wheel', '4-wheel', 'priority']

        count = np.array([0, 0, 0])

        nb_cols = box_and_score.shape[1]

        number_of_rect = len(box_and_score)
        for i in range(number_of_rect):
            if nb_cols > 2:
                # model_result is fully bounding box result
                left, top, right, bottom, score, predicted_class = box_and_score[i, :]

                top = np.floor((top / 512) * im_h).astype('int32')
                left = np.floor((left / 512) * im_w).astype('int32')
                bottom = np.floor((bottom / 512) * im_h).astype('int32')
                right = np.floor((right / 512) * im_w).astype('int32')

                predicted_class = int(predicted_class)
                label = '%s %.2f' % (label_map[predicted_class], score)

                count[predicted_class] += 1
                # visualize boxes
                # print(top, left, right, bottom)
                if config['show_bounding_box']:
                    cv2.rectangle(left_img, (left, top), (right, bottom), color_scheme[predicted_class], 1)
                    cv2.putText(left_img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color_scheme[predicted_class], 1, cv2.LINE_AA)

                boxes.append([left, top, right - left, bottom - top])
                scores.append(score)

            else:
                score, predicted_class = box_and_score[i, :]
                predicted_class = int(predicted_class)

                count[predicted_class] += 1
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

        cv2.rectangle(left_img, (1, 1), (im_w - 1, im_h - 1), (0, 255, 0), 3)

        result = {
            'label': str(end_label),
            'label_name': end_label_map[end_label],
            'total_count': str(np.sum(count)),
            'count_score': str(compute_count_score(count)),
            'count': str(count.tolist()),
            'count_label_colors': str([bgr_to_hex(color_scheme[i]) for i in range(len(count))]),
            'fps': str(min(fps, cls_fps, dt_fps)),
            'cls_fps': str(cls_fps),
            'dt_fps': str(dt_fps)
        }


    else:
        diff_rate = model_result['diff_rate']

        # cv2.rectangle(img, (1, 1), (output_size[0]-1, 50), (0, 0, 0), -1)

        # cv2.putText(img, 'FPS: %.2f (cls: %.2f, bs: %.2f)' % (fps, cls_fps, bs_fps), (12, 30), cv2.FONT_HERSHEY_SIMPLEX ,
        #               0.5, (255,255,255), 1, cv2.LINE_AA)

        # cv2.putText(img, 'Moving rate: %.2f%%' % diff_rate, (output_size[0]-166, 23), cv2.FONT_HERSHEY_SIMPLEX ,
        #               0.5, (255,255,255), 1, cv2.LINE_AA)

        # cv2.putText(img, f'Label: {end_label_map[end_label]}', (output_size[0]-219, 42), cv2.FONT_HERSHEY_SIMPLEX ,
        #               0.5, (255,255,255), 1, cv2.LINE_AA)

        cv2.rectangle(img, (1, 1), (im_w - 1, im_h - 1), (0, 0, 255), 3)

        result = {
            'label': str(end_label),
            'label_name': end_label_map[end_label],
            'diff_rate': str(diff_rate),
            'fps': str(min(fps, cls_fps, bs_fps)),
            'cls_fps': str(cls_fps),
            'bs_fps': str(bs_fps)
        }

    if display == True:
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.set_axis_off()
        ax.imshow(left_img)

    if config['show_diff_mask'] or config['show_foreground_mask']:

        sub_im_w = int(src_im_w / 2)

        img = np.concatenate((left_img, right_img), axis=1)

        padding = np.zeros((int((src_im_h-im_h) / 2), src_im_w, 3))

        img = np.concatenate((padding, img), axis=0)
        img = np.concatenate((img, padding), axis=0)

    else:
        img = left_img
    return img, result
