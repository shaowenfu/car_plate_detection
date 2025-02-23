# 基于计算机视觉的车牌定位系统

<div align="center">

![车牌定位](https://img.shields.io/badge/CV-车牌定位-blue)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)](https://github.com/ultralytics/yolov8)
[![许可证](https://img.shields.io/badge/许可证-MIT-red.svg)](LICENSE)

简体中文 | [English](README.md)

</div>

## 🚀 项目概述

本项目实现了一个基于多种计算机视觉方法的车牌定位系统，包括传统图像处理方法和深度学习方法。系统采用多方法并行处理策略，以实现在各种场景下的稳健性能。

### ✨ 功能特点

- 多方法并行处理策略
- 支持复杂场景下的车牌定位
- 包含完整的评估指标
- 模块化和可扩展的架构

## 🛠️ 检测方法

| 方法 | 描述 | 优势 | 局限性 |
|------|------|------|--------|
| HSV颜色 | 使用HSV颜色空间进行车牌检测 | 速度快，对光照鲁棒 | 仅限蓝色车牌 |
| Sobel边缘 | 基于边缘检测的方法 | 与颜色无关 | 对噪声敏感 |
| YOLOv8 | 基于深度学习的检测 | 准确率高 | 训练需要GPU |

## 📁 项目结构

```
car_plate_detection/
├── data/                          # 数据目录
│   ├── car2024/                   # 原始数据集
│   ├── processed/                 # 预处理后的数据
│   └── results/                   # 实验结果
├── src/                           # 源代码目录
│   ├── config/                    # 配置文件
│   ├── preprocessing/             # 预处理模块
│   ├── methods/                   # 检测方法
│   │   ├── color_based/          
│   │   ├── edge_based/           
│   │   └── deep_learning/        
│   ├── postprocessing/           # 后处理模块
│   └── utils/                    # 工具函数
├── experiments.ipynb             # 实验记录
├── main.py                       # 主程序
└── requirements.txt              # 项目依赖
```

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
OpenCV >= 4.5
PyTorch >= 1.7
```

### 安装说明

```bash
# 克隆仓库
git clone https://github.com/username/car_plate_detection.git

# 安装依赖
pip install -r requirements.txt
```

### 使用示例

```python
from src.methods.deep_learning.yolo_detector import YOLODetector

# 初始化检测器
detector = YOLODetector()

# 检测车牌
boxes = detector.detect(image)
```

## 📊 性能指标

| 方法 | 准确率 | 召回率 | F1分数 |
|------|--------|--------|--------|
| HSV | 0.92 | 0.88 | 0.90 |
| Sobel | 0.89 | 0.85 | 0.87 |
| YOLOv8 | 0.98 | 0.97 | 0.975 |
| 融合方法 | 0.99 | 0.98 | 0.985 |

## 🔍 Sobel边缘检测改进

### 主要改进点
1. **预处理优化**
   - 使用高斯滤波(3x3)减少图像噪声
   - 添加CLAHE增强对比度
   - 使车牌边缘更加清晰，减少干扰

2. **边缘检测改进**
   - 使用固定大小的Sobel算子(ksize=3)
   - 分别计算x和y方向的梯度
   - 通过归一化处理梯度幅值

3. **二值化处理优化**
   - 使用自适应阈值
   - 参数设置：blockSize=11, C=2
   - 更好地适应不同光照条件

4. **形态学操作优化**
   - 使用矩形结构元素(17x3)
   - 先开运算去除噪点，再闭运算连接边缘
   - 结构元素大小经过反复测试确定

5. **候选区域筛选改进**
   - 增加了基于车牌特征的过滤条件：
     * 面积阈值 > 1000
     * 宽高比范围：2.0-5.0
   - 使用minAreaRect获取最小外接矩形

## 📈 未来改进

1. **参数自适应**
   - 根据图像分辨率自动调整面积阈值
   - 添加基于图像统计特征的自适应参数调整

2. **多尺度处理**
   - 在不同尺度下进行检测
   - 合并多尺度结果提高检测准确率

3. **特征融合**
   - 结合颜色特征和边缘特征
   - 考虑添加纹理特征分析

4. **深度学习结合**
   - 使用深度学习模型对候选区域进行二次验证
   - 考虑采用深度学习的边缘检测方法

## 📝 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request。

## 📧 联系方式

如有问题和反馈，请开启issue或联系shaowenfu.pg@gmail.com。 