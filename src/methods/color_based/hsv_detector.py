import cv2
import numpy as np
from ...config.config import Config

class HSVDetector:
    def __init__(self):
        self.lower = Config.HSV_BLUE_LOWER
        self.upper = Config.HSV_BLUE_UPPER
    
    def detect(self, image):
        """
        基于HSV颜色空间的车牌定位
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建掩码
        mask = cv2.inRange(hsv, self.lower, self.upper)
        
        # 形态学处理
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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