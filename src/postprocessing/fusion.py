import numpy as np
from typing import Dict, Tuple, List

class ResultFusion:
    def weighted_fusion(self, detections: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        对多个检测器的结果进行加权融合
        
        Args:
            detections: 字典，键为检测器名称，值为(boxes, scores)元组
                boxes: shape为(N, 4)的数组，每行为[x1, y1, x2, y2]
                scores: shape为(N,)的数组，表示每个框的置信度
        """
        if not detections:
            return np.array([])
            
        # 收集所有检测框和分数
        all_boxes = []
        all_scores = []
        
        for method, (boxes, scores) in detections.items():
            if len(boxes) == 0:
                continue
                
            # 确保boxes是2D数组
            boxes = np.asarray(boxes, dtype=np.float32)
            if boxes.ndim == 1:
                boxes = boxes.reshape(1, -1)
            
            # 确保scores是1D数组
            scores = np.asarray(scores, dtype=np.float32)
            if scores.ndim == 2:
                scores = scores.reshape(-1)
                
            all_boxes.append(boxes)
            all_scores.append(scores)
        
        if not all_boxes:
            return np.array([])
            
        # 合并所有检测框和分数
        boxes = np.vstack(all_boxes)
        scores = np.hstack(all_scores)
        
        # 应用NMS
        return self.nms(boxes, scores)
    
    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
        """
        非极大值抑制
        """
        if len(boxes) == 0:
            return np.array([])
            
        # 获取框的坐标
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # 计算框的面积
        areas = (x2 - x1) * (y2 - y1)
        
        # 按分数排序
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
                
            # 计算IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            # 获取重叠度小于阈值的索引
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        
        return boxes[keep]
