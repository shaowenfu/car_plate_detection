import numpy as np
from ..utils.metrics import calculate_iou

class ResultFusion:
    def __init__(self, iou_threshold=0.5):
        """
        多方法结果融合
        
        Args:
            iou_threshold: IoU阈值
        """
        self.iou_threshold = iou_threshold
        
    def nms(self, boxes, scores):
        """
        非极大值抑制
        """
        if len(boxes) == 0:
            return []
            
        # 转换为numpy数组
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # 按得分排序
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while indices.size > 0:
            current = indices[0]
            keep.append(current)
            
            if indices.size == 1:
                break
                
            ious = np.array([calculate_iou(boxes[current], boxes[i]) 
                           for i in indices[1:]])
            
            indices = indices[1:][ious < self.iou_threshold]
            
        return boxes[keep].tolist()
        
    def weighted_fusion(self, detections):
        """
        加权融合多个检测器的结果
        
        Args:
            detections: 字典，键为检测器名称，值为(boxes, scores)元组
        """
        all_boxes = []
        all_scores = []
        
        # 合并所有检测结果
        for detector_name, (boxes, scores) in detections.items():
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            
        # 应用NMS得到最终结果
        final_boxes = self.nms(all_boxes, all_scores)
        return final_boxes
