import torch
import torch.nn as nn
from ...config.config import Config
import os

class YOLODetector:
    def __init__(self, model_path=None):
        """
        基于YOLOv5的车牌检测器
        
        Args:
            model_path: 预训练模型路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """
        加载YOLOv5模型
        """
        if model_path and os.path.exists(model_path):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model.to(self.device)
        return model
        
    def detect(self, image):
        """
        检测图像中的车牌
        
        Args:
            image: 输入图像
            
        Returns:
            list: 检测到的车牌边界框列表
        """
        results = self.model(image)
        boxes = []
        
        # 处理检测结果
        for det in results.xyxy[0]:
            if det[5] == 0:  # 假设车牌类别索引为0
                box = det[:4].cpu().numpy()
                boxes.append(box)
                
        return boxes
