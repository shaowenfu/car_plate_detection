# 基于计算机视觉的车牌定位系统

本项目实现了一个基于多种计算机视觉方法的车牌定位系统，包括传统图像处理方法和深度学习方法。

## 功能特点

- 多方法并行处理策略
- 支持复杂场景下的车牌定位
- 包含完整的评估指标

## 项目结构 
```
car_plate_detection/
├── data/                          # 数据目录
│   ├── car2024/                   # 原始数据集
│   ├── processed/                 # 预处理后的数据
│   └── results/                   # 实验结果
├── src/                           # 源代码目录
│   ├── config/                    # 配置文件
│   │   ├── __init__.py
│   │   └── config.py             # 参数配置
│   ├── preprocessing/             # 预处理模块
│   │   ├── __init__.py
│   │   ├── noise_reduction.py    # 降噪处理
│   │   ├── illumination.py       # 光照均衡化
│   │   └── resize.py             # 图像尺寸标准化
│   ├── methods/                   # 定位方法
│   │   ├── __init__.py
│   │   ├── color_based/          # 基于颜色的方法
│   │   │   ├── __init__.py
│   │   │   └── hsv_detector.py
│   │   ├── edge_based/           # 基于边缘的方法
│   │   │   ├── __init__.py
│   │   │   └── sobel_detector.py
│   │   └── deep_learning/        # 深度学习方法
│   │       ├── __init__.py
│   │       ├── dataset.py
│   │       └── yolo_detector.py
│   ├── postprocessing/           # 后处理模块
│   │   ├── __init__.py
│   │   ├── filter.py            # 候选区域筛选
│   │   └── fusion.py            # 结果融合
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       ├── visualization.py      # 可视化工具
│       └── metrics.py           # 评估指标
├── experiments.ipynb            # 实验记录
├── main.py                      # 主程序
├── requirements.txt             # 项目依赖
├── README.md                    # 项目说明
├── report.md                    # 实验报告
└── .gitignore                  # Git忽略文件
```