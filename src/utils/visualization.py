import cv2
import numpy as np

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制检测框
    """
    img = image.copy()
    for box in boxes:
        cv2.drawContours(img, [box], 0, color, thickness)
    return img

def show_results(original, processed, title='Result'):
    """
    显示原图和处理后的图像对比
    """
    combined = np.hstack((original, processed))
    cv2.imshow(title, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 