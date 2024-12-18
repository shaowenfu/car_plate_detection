import cv2
import numpy as np

def histogram_equalization(image):
    """
    直方图均衡化处理
    """
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return cv2.equalizeHist(image)

def adaptive_histogram_equalization(image):
    """
    自适应直方图均衡化
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return clahe.apply(image) 