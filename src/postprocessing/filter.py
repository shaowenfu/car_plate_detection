import cv2
import numpy as np
from ..config.config import Config

class CandidateFilter:
    def __init__(self):
        self.min_area = Config.MIN_PLATE_AREA
        self.max_area = Config.MAX_PLATE_AREA
        self.min_ratio = Config.MIN_ASPECT_RATIO
        self.max_ratio = Config.MAX_ASPECT_RATIO
        
    def filter_by_area(self, candidates):
        """
        根据面积过滤候选区域
        """
        filtered = []
        for box in candidates:
            area = cv2.contourArea(box)
            if self.min_area < area < self.max_area:
                filtered.append(box)
        return filtered
        
    def filter_by_ratio(self, candidates):
        """
        根据宽高比过滤候选区域
        """
        filtered = []
        for box in candidates:
            rect = cv2.minAreaRect(box)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            ratio = max(w, h) / min(w, h)
            if self.min_ratio < ratio < self.max_ratio:
                filtered.append(box)
        return filtered
        
    def apply_filters(self, candidates):
        """
        应用所有过滤器
        """
        candidates = self.filter_by_area(candidates)
        candidates = self.filter_by_ratio(candidates)
        return candidates
