import cv2
import numpy as np

from inference.models.frame_difference import FrameDiffEstimator


# cap = cv2.VideoCapture(r'C:\Users\nvdkg\Documents\Camtasia\dnnew_crowd.autosave\dnnew_crowd.autosave.mp4')
# cap = cv2.VideoCapture(r'C:\Users\nvdkg\Documents\Camtasia\test3_crowd\test3_crowd.mp4')
cap = cv2.VideoCapture(r"D:\dnnew_crowd.mp4")

frame_diff_estimator = FrameDiffEstimator(debug=True)

while True:
    _, frame = cap.read()

    frame = cv2.resize(frame, (512, 512))
    
    cv2.imshow("Frame0", frame)
    cv2.imshow("Frame0.", frame)
    
    frame_diff_estimator.apply(frame)
    
    # cv2.imshow("Contours", image_copy)
    # cv2.imshow("Mask Contours", mask_copy)

    # cv2.imshow("diff_mask", diff_mask)
    # cv2.imshow("current_foreground", current_foreground)

    # print(np.sum(diff_mask > 0) / np.sum(current_foreground > 0))

    # cv2.imshow("Diff Mask", diff_mask)

    key = cv2.waitKey(30)
    if key == 27:
        break
    elif key == 32:
        key1 = 0
        while key1 != 32:
            key1 = cv2.waitKey()


cap.release()
cv2.destroyAllWindows()
