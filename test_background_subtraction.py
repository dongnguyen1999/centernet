import cv2
import numpy as np

from utils.limit_queue import LimitQueue

queue_size = 5
frame_queue = LimitQueue(queue_size)
bs_threshold = 30
median_blur_kernel = 7

# cap = cv2.VideoCapture(r'C:\Users\nvdkg\Documents\Camtasia\dnnew_crowd.autosave\dnnew_crowd.autosave.mp4')
# cap = cv2.VideoCapture(r'C:\Users\nvdkg\Documents\Camtasia\test3_crowd\test3_crowd.mp4')
cap = cv2.VideoCapture(r"D:\dnnew_crowd.mp4")

_, first_frame = cap.read()
# gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)

# gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold=bs_threshold, detectShadows=False)
# subtractor = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=1000)



while True:
    _, frame = cap.read()
    frame_queue.enqueue(frame)

    prev_frame = frame_queue.top()
    prev_frame = cv2.resize(prev_frame, (512, 512))
    frame = cv2.resize(frame, (512, 512))

    # prev_mask = subtractor.apply(prev_frame)
    mask = subtractor.apply(frame)

    # prev_mask = cv2.medianBlur(prev_mask, median_blur_kernel)
    mask = cv2.medianBlur(mask, median_blur_kernel)
    
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # difference = cv2.absdiff(gray_frame, frame)
    # _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

    # cv2.imshow("First Frame", gray_frame)
    cv2.imshow("Frame", frame)
    # cv2.imshow("Difference", difference)

    ret, thresh = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # draw contours on the original image
    image_copy = frame.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                

    cv2.imshow("Mask", mask)
    cv2.imshow("Contours", image_copy)


    # print(np.sum(mask > 0) / (mask.shape[0]*mask.shape[1]))

    # cv2.imshow("Diff Mask", mask-prev_mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
