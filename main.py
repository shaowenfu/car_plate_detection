import cv2
import os
import numpy as np
from src.config.config import Config
from src.preprocessing.noise_reduction import gaussian_filter, median_filter
from src.preprocessing.illumination import histogram_equalization, adaptive_histogram_equalization
from src.methods.color_based.hsv_detector import HSVDetector
from src.methods.edge_based.sobel_detector import SobelDetector
from src.methods.deep_learning.yolo_detector import YOLODetector
from src.postprocessing.filter import CandidateFilter
from src.postprocessing.fusion import ResultFusion
from src.utils.visualization import draw_boxes

def train_yolo():
    """训练YOLO模型"""
    print("开始训练YOLO模型...")
    
    # 初始化检测器
    detector = YOLODetector()
    
    # 使用K折交叉验证训练模型
    metrics = detector.train_with_kfold('data', k=5)
    
    print("YOLO模型训练完成")
    # print("平均性能指标:", metrics)
    return detector

def process_single_image(image_path, detector, debug=False):
    """处理单张图片"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None, None
    
    # 预处理
    processed_image = gaussian_filter(image)
    processed_image = adaptive_histogram_equalization(processed_image)
    
    # 多方法检测
    detections = {}
    all_results = {}
    
    try:
        # 颜色检测
        print("开始HSV检测...")
        hsv_detector = HSVDetector()
        hsv_boxes = hsv_detector.detect(processed_image)
        if debug:
            print("HSV检测结果:")
            print(f"检测到 {len(hsv_boxes)} 个目标")
            if hsv_boxes:
                hsv_result = draw_boxes(processed_image.copy(), hsv_boxes)
                all_results['hsv'] = hsv_result
        if hsv_boxes:
            detections['hsv'] = (hsv_boxes, [0.8] * len(hsv_boxes))
        
        # 边缘检测
        print("开始Sobel检测...")
        sobel_detector = SobelDetector()
        sobel_boxes = sobel_detector.detect(processed_image)
        if debug:
            print("Sobel检测结果:")
            print(f"检测到 {len(sobel_boxes)} 个目标")
            if sobel_boxes:
                sobel_result = draw_boxes(processed_image.copy(), sobel_boxes)
                all_results['sobel'] = sobel_result
        if sobel_boxes:
            detections['sobel'] = (sobel_boxes, [0.7] * len(sobel_boxes))
        
        # YOLO检测
        print("开始YOLO检测...")
        yolo_boxes = detector.detect(processed_image)
        if debug:
            print("YOLO检测结果:")
            print(f"检测到 {len(yolo_boxes)} 个目标")
            if yolo_boxes:
                yolo_result = draw_boxes(processed_image.copy(), yolo_boxes)
                all_results['yolo'] = yolo_result
        if yolo_boxes:
            # 确保confidence scores的长度与boxes匹配
            confidence_scores = [0.9] * len(yolo_boxes)
            detections['yolo'] = (yolo_boxes, confidence_scores)
        
        # 结果融合
        if not detections:
            print(f"未检测到任何目标: {image_path}")
            return processed_image, None
            
        fusion = ResultFusion()
        try:
            # 确保所有检测结果的格式一致
            formatted_detections = {}
            for method, (boxes, scores) in detections.items():
                if len(boxes) > 0:
                    # 确保boxes和scores是numpy数组且维度正确
                    boxes = np.array(boxes, dtype=np.float32)
                    scores = np.array(scores, dtype=np.float32)
                    if boxes.ndim == 1:
                        boxes = boxes.reshape(1, -1)
                    formatted_detections[method] = (boxes, scores)
            
            final_boxes = fusion.weighted_fusion(formatted_detections)

            if final_boxes is None or len(final_boxes) == 0:
                print("融合结果为空，使用单个检测器结果")
                # 按优先级尝试不同检测器的结果
                for method in ['yolo', 'hsv', 'sobel']:
                    if method in detections and len(detections[method][0]) > 0:
                        final_boxes = detections[method][0]
                        break
                if not final_boxes or len(final_boxes) == 0:
                    return processed_image, None


        except Exception as e:
            print(f"NMS处理错误: {str(e)}")
            # 按优先级使用单个检测器的结果
            for method in ['yolo', 'hsv', 'sobel']:
                if method in detections and len(detections[method][0]) > 0:
                    final_boxes = detections[method][0]
                    break
            if not final_boxes:
                return processed_image, None
        
        if debug:
            print("\n融合后的结果:")
            print(f"最终检测到 {len(final_boxes)} 个目标")
        
        # 后处理
        candidate_filter = CandidateFilter()
        try:    
            final_boxes = candidate_filter.apply_filters(final_boxes)
        except Exception as e:
            print(f"后处理错误: {str(e)}")
            return processed_image, None
        
        # 可视化结果
        result_image = draw_boxes(processed_image.copy(), final_boxes)
        
        return result_image, all_results
        
    except Exception as e:
        print(f"处理图片时出错 {image_path}: {str(e)}")
        return processed_image, None

def main():
    """主函数"""
    # 第一步：验证标注
    verify_annotations()

    # 第二步：训练YOLO模型
    detector = train_yolo()
    
    # 第三步：创建结果目录
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # 第四步：处理测试图片
    image_files = [f for f in os.listdir(Config.TRAIN_DATA_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    count = 0
    for image_file in image_files:
        print(f"开始处理测试图片{count+1}: {image_file}")
        image_path = os.path.join(Config.TRAIN_DATA_DIR, image_file)
        result_image, detector_results = process_single_image(image_path, detector, debug=True)
        
        if result_image is not None and detector_results is not None:
            count += 1
            # 保存结果
            base_name = os.path.splitext(image_file)[0]
            
            # 保存各检测器的结果
            for detector_name, detector_image in detector_results.items():
                result_path = os.path.join(Config.RESULTS_DIR, 
                    f"{base_name}_{detector_name}.jpg")
                cv2.imwrite(result_path, detector_image)
                print(f"保存 {detector_name} 检测结果至: {result_path}")
            
            # 保存融合结果
            fusion_path = os.path.join(Config.RESULTS_DIR, 
                f"{base_name}_fusion.jpg")
            cv2.imwrite(fusion_path, result_image)
            print(f"最终结果已保存至: {fusion_path}")
    
    print(f"处理完成，共处理 {count} 张图片")

def verify_annotations():
    """验证标注完整性"""
    image_dir = 'data/car2024'
    label_dir = 'data/labels'
    
    # 获取所有图片和标注文件
    images = set(f.rsplit('.', 1)[0] for f in os.listdir(image_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    labels = set(f.rsplit('.', 1)[0] for f in os.listdir(label_dir)
                if f.endswith('.txt'))
    
    # 检查是否每张图片都有对应的标注
    unlabeled = images - labels
    if unlabeled:
        print("以下图片未标注:")
        for name in sorted(unlabeled):
            print(f"  {name}")
    else:
        print("所有图片都已标注!")
    
    # 验证标注格式
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
            
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            try:
                # YOLO格式: class_id x_center y_center width height
                values = [float(x) for x in line.strip().split()]
                if len(values) != 5:
                    print(f"格式错误 {label_file} 第{i+1}行: 需要5个值")
                if not (0 <= values[1] <= 1 and 0 <= values[2] <= 1 and
                       0 <= values[3] <= 1 and 0 <= values[4] <= 1):
                    print(f"坐标错误 {label_file} 第{i+1}行: 值必须在0-1之间")
            except:
                print(f"解析错误 {label_file} 第{i+1}行")

if __name__ == "__main__":
    main() 