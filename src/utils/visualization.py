import cv2
import numpy as np

def draw_boxes(image, boxes, color=(0, 165, 255), thickness=10):
    """
    在图像上绘制边界框
    
    Args:
        image: 输入图像
        boxes: 边界框列表，每个边界框为 [x1, y1, x2, y2] 格式
        color: BGR颜色元组，默认橙色 (0,165,255)
        thickness: 线条粗细，默认5
    """
    try:
        # 获取图像尺寸
        height, width = image.shape[:2]
        
        # 创建图像副本
        result = image.copy()
        
        # 绘制每个边界框
        for box in boxes:
            # 确保坐标在图像范围内
            x1 = max(0, min(int(box[0]), width-1))
            y1 = max(0, min(int(box[1]), height-1))
            x2 = max(0, min(int(box[2]), width-1))
            y2 = max(0, min(int(box[3]), height-1))
            
            # 绘制矩形
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        return result
    except Exception as e:
        print(f"绘制边界框时出错: {str(e)}")
        return image

def show_results(image, title='Result'):
    """
    显示图像结果
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 