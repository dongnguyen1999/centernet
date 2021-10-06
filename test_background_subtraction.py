import cv2
import numpy as np

cap = cv2.VideoCapture(r'C:\Users\nvdkg\Documents\Camtasia\dnnew_crowd.autosave\dnnew_crowd.autosave.mp4')
# cap = cv2.VideoCapture(r'C:\Users\nvdkg\Documents\Camtasia\test3_crowd\test3_crowd.mp4')

_, first_frame = cap.read()
gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

subtractor = cv2.createBackgroundSubtractorMOG2(history=15, varThreshold=100, detectShadows=False)
# subtractor = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=1000)
while True:
    _, frame = cap.read()

    mask = subtractor.apply(frame)
    
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # difference = cv2.absdiff(gray_frame, frame)
    # _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

    # cv2.imshow("First Frame", gray_frame)
    cv2.imshow("Frame", frame)
    # cv2.imshow("Difference", difference)
    cv2.imshow("Mask", cv2.medianBlur(mask, 1))
    print(np.sum(mask > 0) / (mask.shape[0]*mask.shape[1]))

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
