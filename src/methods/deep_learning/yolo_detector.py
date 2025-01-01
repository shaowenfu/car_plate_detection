from ultralytics import YOLO
import torch
import os
import cv2
from sklearn.model_selection import KFold
import numpy as np
import shutil
import matplotlib.pyplot as plt
from datetime import datetime

class YOLODetector:
    def __init__(self, weights_path=None):
        """
        基于YOLOv8的车牌检测器
        
        Args:
            weights_path: 训练后的权重文件路径，如果为None则使用预训练模型
        """
        self.device = 'cpu'  # 强制使用CPU
        if weights_path and os.path.exists(weights_path):
            print(f"使用预训练模型: {weights_path}")
            self.model = YOLO(weights_path)
        else:
            print("使用默认预训练模型: yolov8n.pt")
            self.model = YOLO('yolov8n.pt')
            
        self.training_history = {
            'precision': [],
            'recall': [],
            'mAP50': [],
            'mAP50-95': [],
            'loss': []
        }
        
    def plot_metrics(self, save_dir='runs/train/metrics'):
        """绘制训练过程中的性能指标图表"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 准确率和召回率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['precision'], label='Precision')
        plt.plot(self.training_history['recall'], label='Recall')
        plt.title('Precision and Recall over Training')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'precision_recall_{timestamp}.png'))
        plt.close()
        
        # 2. mAP曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['mAP50'], label='mAP@0.5')
        plt.plot(self.training_history['mAP50-95'], label='mAP@0.5:0.95')
        plt.title('mAP Metrics over Training')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'mAP_{timestamp}.png'))
        plt.close()
        
        # 3. 损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['loss'], label='Total Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'loss_{timestamp}.png'))
        plt.close()
        
        # 4. 生成性能报告
        report_path = os.path.join(save_dir, f'training_report_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write("YOLO Training Performance Report\n")
            f.write("=" * 30 + "\n\n")
            
            f.write("Final Metrics:\n")
            f.write("-" * 15 + "\n")
            for metric in ['precision', 'recall', 'mAP50', 'mAP50-95']:
                if self.training_history[metric]:
                    final_value = self.training_history[metric][-1]
                    f.write(f"{metric}: {final_value:.4f}\n")
            
            f.write("\nBest Metrics:\n")
            f.write("-" * 15 + "\n")
            for metric in ['precision', 'recall', 'mAP50', 'mAP50-95']:
                if self.training_history[metric]:
                    best_value = max(self.training_history[metric])
                    best_epoch = self.training_history[metric].index(best_value) + 1
                    f.write(f"Best {metric}: {best_value:.4f} (Epoch {best_epoch})\n")
    
    def train_with_kfold(self, data_dir, k=5):
        """使用K折交叉验证训练模型"""
        # 转换为绝对路径
        data_dir = os.path.abspath(data_dir)
        
        # 获取所有图片和标注文件
        image_files = sorted([f for f in os.listdir(os.path.join(data_dir, 'car2024'))
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # 在训练前预处理图片
        def check_and_fix_images(image_files, src_dir):
            for img_name in image_files:
                img_path = os.path.join(src_dir, img_name)
                try:
                    # 尝试读取图片
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: 无法读取图片 {img_path}")
                        continue
                    # 重新保存图片
                    cv2.imwrite(img_path, img)
                except Exception as e:
                    print(f"处理图片时出错 {img_path}: {str(e)}")
        
        # 预处理所有图片
        check_and_fix_images(image_files, os.path.join(data_dir, 'car2024'))
        
        # 创建K折交叉验证器
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        # 记录每折的性能
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
            print(f"\n开始训练第 {fold + 1} 折...")
            
            # 创建当前折的数据集目录
            fold_dir = os.path.join(data_dir, f'fold_{fold}')
            train_img_dir = os.path.join(fold_dir, 'images', 'train')
            val_img_dir = os.path.join(fold_dir, 'images', 'val')
            train_label_dir = os.path.join(fold_dir, 'labels', 'train')
            val_label_dir = os.path.join(fold_dir, 'labels', 'val')
            
            # 创建目录
            os.makedirs(train_img_dir, exist_ok=True)
            os.makedirs(val_img_dir, exist_ok=True)
            os.makedirs(train_label_dir, exist_ok=True)
            os.makedirs(val_label_dir, exist_ok=True)
            
            # 分配文件到训练集和验证集
            for idx in train_idx:
                img_name = image_files[idx]
                label_name = img_name.rsplit('.', 1)[0] + '.txt'
                
                # 复制图片
                shutil.copy2(os.path.join(data_dir, 'car2024', img_name),
                          os.path.join(train_img_dir, img_name))
                # 复制标注
                label_path = os.path.join(data_dir, 'labels', label_name)
                if os.path.exists(label_path):
                    shutil.copy2(label_path, os.path.join(train_label_dir, label_name))
                
            for idx in val_idx:
                img_name = image_files[idx]
                label_name = img_name.rsplit('.', 1)[0] + '.txt'
                
                # 复制图片
                shutil.copy2(os.path.join(data_dir, 'car2024', img_name),
                          os.path.join(val_img_dir, img_name))
                # 复制标注
                label_path = os.path.join(data_dir, 'labels', label_name)
                if os.path.exists(label_path):
                    shutil.copy2(label_path, os.path.join(val_label_dir, label_name))
            
            # 创建当前折的数据集配置文件
            yaml_content = f"""
train: ./images/train  # 训练集图片目录
val: ./images/val      # 验证集图片目录

nc: 1  # 类别数量
names: ['plate']  # 类别名称
"""
            
            yaml_path = os.path.join(fold_dir, 'dataset.yaml')
            with open(yaml_path, 'w') as f:
                f.write(yaml_content.lstrip())
            
            # 修改训练参数，移除不支持的参数
            train_args = {
                'data': yaml_path,
                'epochs': 50,
                'imgsz': 640,
                'batch': 8,
                'device': self.device,
                'patience': 10,
                'save': True,
                'project': 'runs/train',
                'name': f'fold_{fold}',
                'exist_ok': True,
                'verbose': True,  # 启用详细输出
                'save_period': 5  # 每5个epoch保存一次模型
            }
            
            try:
                # 切换到fold目录
                original_dir = os.getcwd()
                os.chdir(fold_dir)
                
                # 训练并记录性能指标
                results = self.model.train(**train_args)
                
                # 从训练结果中提取性能指标
                if hasattr(results, 'results_dict'):
                    metrics = results.results_dict
                    self.training_history['precision'].append(
                        metrics.get('metrics/precision(B)', 0))
                    self.training_history['recall'].append(
                        metrics.get('metrics/recall(B)', 0))
                    self.training_history['mAP50'].append(
                        metrics.get('metrics/mAP50(B)', 0))
                    self.training_history['mAP50-95'].append(
                        metrics.get('metrics/mAP50-95(B)', 0))
                    self.training_history['loss'].append(
                        metrics.get('train/box_loss', 0))
                
                fold_metrics.append(results)
                
                # 绘制当前折的性能图表
                self.plot_metrics(save_dir=os.path.join('runs/train', f'fold_{fold}', 'metrics'))
                
                # 恢复原始目录
                os.chdir(original_dir)
                
            except Exception as e:
                print(f"训练第 {fold + 1} 折时出错: {str(e)}")
                os.chdir(original_dir)
                continue
                
        # 计算并显示平均性能
        if fold_metrics:
            avg_metrics = self._calculate_average_metrics(fold_metrics)
            print("\n交叉验证平均性能:")
            for metric, value in avg_metrics.items():
                print(f"{metric}: {value:.4f}")
                
            # 绘制最终的性能图表
            self.plot_metrics(save_dir='runs/train/final_metrics')
            
            return avg_metrics
        else:
            print("训练失败，没有可用的性能指标")
            return None
    
    def _calculate_average_metrics(self, fold_metrics):
        """计算K折交叉验证的平均性能"""
        metrics_sum = {
            'precision': 0,
            'recall': 0,
            'mAP50': 0,
            'mAP50-95': 0
        }
        
        for metrics in fold_metrics:
            # 获取结果
            results = metrics.results_dict
            metrics_sum['precision'] += results['metrics/precision(B)']  # box precision
            metrics_sum['recall'] += results['metrics/recall(B)']      # box recall
            metrics_sum['mAP50'] += results['metrics/mAP50(B)']       # box mAP@0.5
            metrics_sum['mAP50-95'] += results['metrics/mAP50-95(B)'] # box mAP@0.5:0.95
        
        # 计算平均值
        n = len(fold_metrics)
        return {k: v / n for k, v in metrics_sum.items()}
        
    def detect(self, image):
        """
        检测图像中的车牌
        
        Args:
            image: 输入图像
        
        Returns:
            list: 检测到的车牌边界框列表，每个边界框为[x1, y1, x2, y2]格式
        """
        try:
            results = self.model(image, device=self.device)
            boxes = []
            
            for result in results:
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    for box in result.boxes:
                        if box.conf > 0.5:  # 置信度阈值
                            try:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                box_coords = [
                                    max(0, int(x1)), 
                                    max(0, int(y1)),
                                    min(image.shape[1], int(x2)), 
                                    min(image.shape[0], int(y2))
                                ]
                                boxes.append(box_coords)
                            except Exception as e:
                                print(f"处理检测框时出错: {str(e)}")
                                continue
            
            if not boxes:
                print("未检测到任何车牌")
                return []
            
            return boxes
            
        except Exception as e:
            print(f"YOLO检测出错: {str(e)}")
            return []
