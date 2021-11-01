import cv2
import numpy as np

from utils.limit_queue import LimitQueue

queue_size = 5
frame_queue = LimitQueue(queue_size)

# cap = cv2.VideoCapture(r'C:\Users\nvdkg\Documents\Camtasia\dnnew_crowd.autosave\dnnew_crowd.autosave.mp4')
# cap = cv2.VideoCapture(r'C:\Users\nvdkg\Documents\Camtasia\test3_crowd\test3_crowd.mp4')
cap = cv2.VideoCapture(r"D:\dnnew_crowd.mp4")

_, first_frame = cap.read()
# gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)

# gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

# subtractor = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=1000)
long_subtractor = cv2.createBackgroundSubtractorMOG2(history=162000, varThreshold=50, detectShadows=False)
short_subtractor = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=50, detectShadows=False)

def preprocess(frame):
    contrast = 3
    brightness = -255
    frame = cv2.addWeighted( frame, contrast, frame, 0, brightness)
    frame = cv2.GaussianBlur(frame, (5, 5), 3)
    return frame

def posprocess(mask):
    mask = cv2.medianBlur(mask, 1)
    return mask

def frame_subtraction(frame):
    frame = cv2.resize(frame, (512, 512))
    source_img = frame.copy()

    frame = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

    mask = short_subtractor.apply(frame)

    source_img[mask > 128] = (0, 255, 0)

    # cv2.imshow("frame_subtraction", source_img)

    return mask

def foreground_estimation(frame):

    frame = cv2.resize(frame, (512, 512))
    source_img = frame.copy()

    frame = preprocess(frame)

    mask = long_subtractor.apply(frame)

    (contours, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image=source_img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

    # cv2.imshow("foreground_detection", source_img)

    mask_copy = mask.copy()

    cv2.drawContours(image=mask_copy, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    return mask_copy

while True:
    _, frame = cap.read()

    frame_queue.enqueue(frame)
    prev_frame = frame_queue.top()
    
    # cv2.imshow("Frame", frame)
    
    diff_mask = frame_subtraction(frame)
    
    current_foreground = foreground_estimation(frame)
    
    # cv2.imshow("Contours", image_copy)
    # cv2.imshow("Mask Contours", mask_copy)

    cv2.imshow("diff_mask", diff_mask)
    cv2.imshow("current_foreground", current_foreground)

    print(np.sum(diff_mask > 0) / np.sum(current_foreground > 0))

    # cv2.imshow("Diff Mask", diff_mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
