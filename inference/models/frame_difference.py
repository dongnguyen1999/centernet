import cv2
import numpy as np
from matplotlib import pyplot as plt

from inference.utils import get_masked_img


class FrameDiffEstimator:
    def __init__(self, long_term_history=162000, short_term_history=1, threshold=70, debug=False):
        self.long_term_history = long_term_history
        self.short_term_history = short_term_history
        self.threshold = threshold
        self.debug = debug
        self.refresh()

    def refresh(self):
        self.long_subtractor = cv2.createBackgroundSubtractorMOG2(history=self.long_term_history,
                                                                  varThreshold=self.threshold, detectShadows=False)
        self.short_subtractor = cv2.createBackgroundSubtractorMOG2(history=self.short_term_history,
                                                                   varThreshold=self.threshold, detectShadows=False)

    def feed(self, frame, mask=None):
        if mask is not None:
            frame = get_masked_img(frame, mask)
        self.long_subtractor.apply(frame)
        self.short_subtractor.apply(frame)

    def preprocess(self, frame):
        # increase contrast in image
        contrast = 3
        brightness = -255
        frame = cv2.addWeighted(frame, contrast, frame, 0, brightness)

        # use gaussian blur to reduce noise
        frame = cv2.GaussianBlur(frame, (5, 5), 3)

        return frame

    def posprocess(self, mask):
        # use median blur to reduce noise in output mask
        mask = cv2.medianBlur(mask, 1)
        return mask

    def frame_subtraction(self, frame):
        frame = cv2.resize(frame, (512, 512))

        source_img = frame.copy()

        frame = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

        mask = self.short_subtractor.apply(frame)

        source_img[mask > 0] = (255, 0, 0)

        if self.debug == True: cv2.imshow("frame_subtraction", source_img)

        return mask

    def foreground_estimation(self, frame):

        frame = cv2.resize(frame, (512, 512))

        source_img = frame.copy()

        frame = self.preprocess(frame)

        mask = self.long_subtractor.apply(frame)

        mask = self.posprocess(mask)

        mask_copy = mask.copy()

        (contours, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.debug == True:
            cv2.drawContours(image=source_img, contours=contours, contourIdx=-1, color=(0, 255, 255), thickness=-1,
                             lineType=cv2.LINE_AA)
            cv2.imshow("foreground_detection", source_img)

        cv2.drawContours(image=mask_copy, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=-1,
                         lineType=cv2.LINE_AA)

        return mask_copy

    def apply(self, frame, mask=None, return_diff_mask=False, return_foreground_mask=False):

        if mask is not None:
            frame = get_masked_img(frame, mask)

        diff_mask = self.frame_subtraction(frame)
        foreground_mask = self.foreground_estimation(frame)

        result = {'diff_rate': (np.sum(diff_mask > 0) * 100) / (np.sum(foreground_mask > 0) + 1e-6)}
        if return_diff_mask:
            result['diff_mask'] = diff_mask
        if return_foreground_mask:
            result['foreground_mask'] = foreground_mask

        return result

    def apply_range(self, frames, mask=None):
        for frame in frames:
            self.feed(frame, mask=mask)
