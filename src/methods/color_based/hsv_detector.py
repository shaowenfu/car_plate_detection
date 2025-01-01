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
        try:
            # 获取图像尺寸
            # shape[:2]返回图像的高度和宽度,忽略通道数
            height, width = image.shape[:2]
            
            # HSV颜色空间转换
            # HSV比RGB更适合进行颜色分割,因为它将颜色的色调(H)、饱和度(S)和亮度(V)分离
            # 车牌蓝色在HSV空间中有相对固定的H值范围,不易受光照影响
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 创建二值掩码
            # cv2.inRange会将HSV图像中在指定范围内的像素设为255(白色),
            # 范围外的像素设为0(黑色),从而实现颜色分割
            mask = cv2.inRange(hsv, self.lower, self.upper)
            
            # 形态学操作用于去除噪点和连接断开的区域
            kernel = np.ones((5,5), np.uint8)
            # 开运算(先腐蚀后膨胀)去除小的噪点
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # 闭运算(先膨胀后腐蚀)填充内部小洞并连接相邻区域
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 寻找轮廓
            # RETR_EXTERNAL只检测最外层轮廓
            # CHAIN_APPROX_SIMPLE只保存轮廓的拐点坐标,节省内存
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            boxes = []
            count = 0
            for contour in contours:
                count += 1
                # 计算面积
                area = cv2.contourArea(contour)
                
                # 过滤小面积
                if area > 5000:  # 可以调整这个阈值
                    x, y, w, h = cv2.boundingRect(contour)
                    # 确保坐标在图像范围内
                    x1 = max(0, min(x, width-1))
                    y1 = max(0, min(y, height-1))
                    x2 = max(0, min(x+w, width-1))
                    y2 = max(0, min(y+h, height-1))
                    boxes.append([x1, y1, x2, y2])
            
            return boxes
        except Exception as e:
            print(f"HSV检测器错误: {str(e)}")
            return []