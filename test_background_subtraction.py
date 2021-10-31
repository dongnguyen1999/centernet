import cv2
import numpy as np

from utils.limit_queue import LimitQueue

queue_size = 2
frame_queue = LimitQueue(queue_size)
bs_threshold = 80
median_blur_kernel = 7

cap = cv2.VideoCapture(r'C:\Users\nvdkg\Documents\Camtasia\dnnew_crowd.autosave\dnnew_crowd.autosave.mp4')
# cap = cv2.VideoCapture(r'C:\Users\nvdkg\Documents\Camtasia\test3_crowd\test3_crowd.mp4')
# cap = cv2.VideoCapture(r"D:\dnnew_crowd.mp4")

_, first_frame = cap.read()
# gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)

# gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

subtractor = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=bs_threshold, detectShadows=False)

# subtractor = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=1000)

def preprocess_frame(frame):
    contrast = 3
    brightness = -255
    frame = cv2.addWeighted( frame, contrast, frame, 0, brightness)
    frame = cv2.GaussianBlur(frame, (5, 5), 3)
    return frame

def postprocess_frame(mask):
    mask = cv2.medianBlur(mask, 1)
    # (contours, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_copy = mask.copy()
    # cv2.drawContours(image=mask_copy, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)            
    return mask_copy

while True:
    _, frame = cap.read()
    frame_queue.enqueue(frame)

    prev_frame = frame_queue.top()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    sub = frame-prev_frame
    th = 80
    sub[sub > th] = 1
    sub[sub <= th] = 0
    cv2.imshow("Subtract", sub)

    
    # frame = cv2.medianBlur(frame, 1)
    frame = preprocess_frame(frame)
    prev_frame = preprocess_frame(prev_frame)

    prev_mask = subtractor.apply(prev_frame)
    mask = subtractor.apply(frame)

    # prev_mask = cv2.medianBlur(prev_mask, median_blur_kernel)
    mask = postprocess_frame(mask)
    prev_mask = postprocess_frame(prev_mask)
    
    
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # difference = cv2.absdiff(gray_frame, frame)
    # _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

    # cv2.imshow("First Frame", gray_frame)
    cv2.imshow("Frame", frame)
    # cv2.imshow("Difference", difference)

    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours on the original image

    image_copy = frame.copy()
    im_h, im_w = image_copy.shape[:2]
    
    (contours, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for contour in contours:
    #     (x, y, w, h) = cv2.boundingRect(contour)
    #     if ((w*h) / (im_w*im_h)) > 0.01:
    #         cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)            


    mask_copy = mask.copy()

    cv2.drawContours(image=mask_copy, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)


    cv2.imshow("Mask", mask)
    # cv2.imshow("Contours", image_copy)
    # cv2.imshow("Mask Contours", mask_copy)

    diff_mask = mask-prev_mask

    print(np.sum(diff_mask > 0) / np.sum(prev_mask > 0))

    # cv2.imshow("Diff Mask", diff_mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
