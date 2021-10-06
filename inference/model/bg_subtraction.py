
import cv2
import numpy as np

class BackgroundSubtractorMOG2:
    def __init__(self, history=15, threshold=100):
        self.history = history
        self.threshold = threshold
        self.refresh()

    def refresh(self):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=self.history, varThreshold=self.threshold, detectShadows=False)

    def feed(self, frame):
        self.subtractor.apply(frame)

    def apply(self, frame, return_diff_rate=False):
        mask = self.subtractor.apply(frame)
        if return_diff_rate == True:
            return mask, (np.sum(mask > 0) / (np.size(mask)))
        return mask
    
    def subtract(self, frame1, frame2, return_diff_rate=False):
        self.subtractor.apply(frame1, frame2)
        mask = self.subtractor.apply(frame2)
        if return_diff_rate == True:
            return mask, (np.sum(mask > 0) / (np.size(mask)))
        return mask

    def apply_range(self, frames):
        for frame in frames:
            self.subtractor.apply(frame)
