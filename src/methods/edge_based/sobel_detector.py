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
        try:
            # 获取图像尺寸
            height, width = image.shape[:2]
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. 使用高斯滤波减少噪声
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # 2. 增强对比度
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # 3. Sobel边缘检测
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # 4. 计算梯度幅值
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            magnitude = np.uint8(255 * magnitude / np.max(magnitude))
            
            # 5. 二值化
            _, binary = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 6. 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 7. 确保二值图像是uint8类型
            binary = np.uint8(binary)
            
            # 8. 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            boxes = []
            for contour in contours:
                # 确保轮廓是有效的numpy数组
                if not isinstance(contour, np.ndarray):
                    continue
                
                # 确保轮廓点数足够
                if len(contour) < 5:
                    continue
                
                # 计算面积
                try:
                    area = cv2.contourArea(np.float32(contour))
                except:
                    continue
                
                # 获取最小外接矩形
                try:
                    rect = cv2.minAreaRect(np.float32(contour))
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    # 计算矩形的宽和高
                    w = rect[1][0]
                    h = rect[1][1]
                    
                    # 确保宽度大于高度
                    if w < h:
                        w, h = h, w
                    
                    # 计算宽高比
                    aspect_ratio = float(w) / h if h > 0 else 0
                    
                    # 根据车牌特征进行过滤
                    if (area > 1000 and  # 面积过滤
                        2.0 <= aspect_ratio <= 5.0):  # 车牌的宽高比通常在2-5之间
                        
                        # 获取外接矩形的四个顶点坐标
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # 确保坐标在图像范围内
                        x1 = max(0, min(x, width-1))
                        y1 = max(0, min(y, height-1))
                        x2 = max(0, min(x+w, width-1))
                        y2 = max(0, min(y+h, height-1))
                        
                        boxes.append([x1, y1, x2, y2])
                except:
                    continue
            
            return boxes
            
        except Exception as e:
            print(f"Sobel检测器错误: {str(e)}")
            return [] 