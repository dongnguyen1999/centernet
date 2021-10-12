
import cv2
import numpy as np
from matplotlib import pyplot as plt

from inference.model.decode import get_masked_img

class BackgroundSubtractorMOG2:
    def __init__(self, history=15, threshold=100):
        self.history = history
        self.threshold = threshold
        self.refresh()

    def refresh(self):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=self.history, varThreshold=self.threshold, detectShadows=False)

    def feed(self, frame):
        self.subtractor.apply(frame)

    def apply(self, frame, return_diff_rate=False, mask=None):
        bs_mask = self.subtractor.apply(frame)
        if return_diff_rate == True:
            if mask is not None:
                bs_mask = get_masked_img(bs_mask, mask)
                return bs_mask, (np.sum(bs_mask > 0) * 100 / (np.size(mask[mask == True])))    
            return bs_mask, (np.sum(bs_mask > 0) * 100 / (np.size(bs_mask)))
        return bs_mask
    
    def subtract(self, frame1, frame2, return_diff_rate=False, mask=None):
        self.subtractor.apply(frame1, frame2)
        bs_mask = self.subtractor.apply(frame2)
        if return_diff_rate == True:
            if mask is not None:
                bs_mask = get_masked_img(bs_mask, mask)
                return bs_mask, (np.sum(bs_mask > 0) * 100 / (np.size(mask[mask == True])))
            return bs_mask, (np.sum(bs_mask > 0) * 100 / (np.size(bs_mask)))
        return bs_mask

    def apply_range(self, frames):
        for frame in frames:
            self.subtractor.apply(frame)
