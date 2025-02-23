# Computer Vision Based License Plate Detection System

<div align="center">

![License Plate Detection](https://img.shields.io/badge/CV-License%20Plate%20Detection-blue)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)](https://github.com/ultralytics/yolov8)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

[简体中文](README_zh.md) | English

</div>

## 🚀 Overview

A comprehensive license plate detection system that combines traditional computer vision methods with deep learning approaches. The system employs a multi-method parallel processing strategy to achieve robust performance across various scenarios.

### ✨ Key Features

- Multi-method parallel processing strategy
- Support for complex scene detection
- Complete evaluation metrics
- Modular and extensible architecture

## 🛠️ Methods

| Method | Description | Advantages | Limitations |
|--------|-------------|------------|-------------|
| HSV Color | Uses HSV color space for plate detection | Fast, robust to lighting | Limited to blue plates |
| Sobel Edge | Edge detection based approach | Color independent | Sensitive to noise |
| YOLOv8 | Deep learning based detection | High accuracy | Requires GPU for training |

## 📁 Project Structure

```
car_plate_detection/
├── data/                          # Data directory
│   ├── car2024/                   # Original dataset
│   ├── processed/                 # Processed data
│   └── results/                   # Experimental results
├── src/                           # Source code
│   ├── config/                    # Configuration files
│   ├── preprocessing/             # Preprocessing modules
│   ├── methods/                   # Detection methods
│   │   ├── color_based/          
│   │   ├── edge_based/           
│   │   └── deep_learning/        
│   ├── postprocessing/           # Post-processing modules
│   └── utils/                    # Utility functions
├── experiments.ipynb             # Experiment notebook
├── main.py                       # Main program
└── requirements.txt              # Dependencies
```

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.8
OpenCV >= 4.5
PyTorch >= 1.7
```

### Installation

```bash
# Clone the repository
git clone https://github.com/username/car_plate_detection.git

# Install dependencies
pip install -r requirements.txt
```

### Usage

```python
from src.methods.deep_learning.yolo_detector import YOLODetector

# Initialize detector
detector = YOLODetector()

# Detect license plates
boxes = detector.detect(image)
```

## 📊 Performance

| Method | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| HSV | 0.92 | 0.88 | 0.90 |
| Sobel | 0.89 | 0.85 | 0.87 |
| YOLOv8 | 0.98 | 0.97 | 0.975 |
| Fusion | 0.99 | 0.98 | 0.985 |

## 🔍 Sobel Edge Detection Improvements

### Key Improvements
1. **Preprocessing Optimization**
   - Gaussian filtering (3x3) for noise reduction
   - CLAHE for contrast enhancement
   - Clearer plate edges, reduced interference

2. **Edge Detection Enhancement**
   - Fixed size Sobel operator (ksize=3)
   - Separate X and Y gradient calculation
   - Gradient magnitude normalization

3. **Binarization Optimization**
   - Adaptive thresholding
   - Parameters: blockSize=11, C=2
   - Better adaptation to varying lighting conditions

4. **Morphological Operations**
   - Rectangular structuring element (17x3)
   - Opening operation followed by closing
   - Optimized element size through testing

5. **Candidate Region Filtering**
   - Enhanced filtering conditions:
     * Area threshold > 1000
     * Aspect ratio range: 2.0-5.0
   - minAreaRect for accurate rectangle fitting

## 📈 Future Improvements

1. **Parameter Adaptation**
   - Auto-adjust area threshold based on image resolution
   - Add adaptive parameter tuning based on image statistics

2. **Multi-scale Processing**
   - Implement detection at different scales
   - Merge multi-scale results

3. **Feature Fusion**
   - Combine color and edge features
   - Add texture feature analysis

4. **Deep Learning Integration**
   - Secondary verification using deep learning
   - Consider deep learning based edge detection

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and feedback, please open an issue or contact shaowenfu.pg@gmail.com.