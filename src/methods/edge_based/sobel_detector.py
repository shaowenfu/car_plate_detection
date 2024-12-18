import cv2
import numpy as np
from ...config.config import Config

class SobelDetector:
    def __init__(self):
        self.kernel_size = Config.SOBEL_KERNEL_SIZE
    
    def detect(self, image):
        """
        基于Sobel算子的车牌定位
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sobel边缘检测
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=self.kernel_size)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=self.kernel_size)
        
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        # 合并梯度
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        # 二值化
        _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学处理
        kernel = np.ones((5,19), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选候选区域
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if Config.MIN_PLATE_AREA < area < Config.MAX_PLATE_AREA:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                candidates.append(box)
        
        return candidates 