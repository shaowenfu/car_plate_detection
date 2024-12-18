import cv2
import numpy as np

def gaussian_filter(image, kernel_size=(5,5), sigma=1.0):
    """
    使用高斯滤波进行图像降噪
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)

def median_filter(image, kernel_size=5):
    """
    使用中值滤波进行图像降噪
    """
    return cv2.medianBlur(image, kernel_size) 