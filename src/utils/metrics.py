import numpy as np

def calculate_iou(box1, box2):
    """
    计算两个框的IoU
    """
    # 将点集转换为矩形表示
    x1 = np.min(box1[:, 0])
    y1 = np.min(box1[:, 1])
    x2 = np.max(box1[:, 0])
    y2 = np.max(box1[:, 1])
    
    x3 = np.min(box2[:, 0])
    y3 = np.min(box2[:, 1])
    x4 = np.max(box2[:, 0])
    y4 = np.max(box2[:, 1])
    
    # 计算交集区域
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def evaluate_detection(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    评估检测结果
    """
    tp = 0
    fp = len(pred_boxes)
    fn = len(gt_boxes)
    
    for pred_box in pred_boxes:
        for gt_box in gt_boxes:
            if calculate_iou(pred_box, gt_box) > iou_threshold:
                tp += 1
                fp -= 1
                fn -= 1
                break
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    } 