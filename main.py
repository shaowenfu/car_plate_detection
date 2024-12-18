import cv2
import os
from src.config.config import Config
from src.preprocessing.noise_reduction import gaussian_filter, median_filter
from src.preprocessing.illumination import histogram_equalization, adaptive_histogram_equalization
from src.methods.color_based.hsv_detector import HSVDetector
from src.methods.edge_based.sobel_detector import SobelDetector
from src.methods.deep_learning.yolo_detector import YOLODetector
from src.postprocessing.filter import CandidateFilter
from src.postprocessing.fusion import ResultFusion
from src.utils.visualization import draw_boxes, show_results

def process_single_image(image_path):
    """
    处理单张图片
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 预处理
    image = gaussian_filter(image)
    image = adaptive_histogram_equalization(image)
    
    # 多方法检测
    detections = {}
    
    # 颜色检测
    hsv_detector = HSVDetector()
    hsv_boxes = hsv_detector.detect(image)
    detections['hsv'] = (hsv_boxes, [0.8] * len(hsv_boxes))  # 示例置信度
    
    # 边缘检测
    sobel_detector = SobelDetector()
    sobel_boxes = sobel_detector.detect(image)
    detections['sobel'] = (sobel_boxes, [0.7] * len(sobel_boxes))  # 示例置信度
    
    # YOLO检测
    yolo_detector = YOLODetector()
    yolo_boxes = yolo_detector.detect(image)
    detections['yolo'] = (yolo_boxes, [0.9] * len(yolo_boxes))  # 示例置信度
    
    # 结果融合
    fusion = ResultFusion()
    final_boxes = fusion.weighted_fusion(detections)
    
    # 后处理
    candidate_filter = CandidateFilter()
    final_boxes = candidate_filter.apply_filters(final_boxes)
    
    # 可视化结果
    result_image = draw_boxes(image, final_boxes)
    
    return result_image

def main():
    """
    主函数
    """
    # 创建结果目录
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # 获取所有测试图片
    image_files = [f for f in os.listdir(Config.TRAIN_DATA_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 处理每张图片
    for image_file in image_files:
        image_path = os.path.join(Config.TRAIN_DATA_DIR, image_file)
        result_image = process_single_image(image_path)
        
        if result_image is not None:
            # 保存结果
            result_path = os.path.join(Config.RESULTS_DIR, f"result_{image_file}")
            cv2.imwrite(result_path, result_image)
            print(f"处理完成: {image_file}")

if __name__ == "__main__":
    main() 