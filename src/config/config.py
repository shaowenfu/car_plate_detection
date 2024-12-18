class Config:
    # 数据相关配置
    DATA_ROOT = "data"
    TRAIN_DATA_DIR = "data/car2024"
    PROCESSED_DATA_DIR = "data/processed"
    RESULTS_DIR = "data/results"
    
    # 预处理参数
    RESIZE_WIDTH = 640
    RESIZE_HEIGHT = 480
    GAUSSIAN_KERNEL_SIZE = (5, 5)
    GAUSSIAN_SIGMA = 1.0
    
    # 颜色检测参数
    HSV_BLUE_LOWER = (100, 43, 46)
    HSV_BLUE_UPPER = (124, 255, 255)
    
    # 边缘检测参数
    SOBEL_KERNEL_SIZE = 3
    CANNY_LOW_THRESHOLD = 50
    CANNY_HIGH_THRESHOLD = 150
    
    # 深度学习参数
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # 后处理参数
    MIN_PLATE_AREA = 2000
    MAX_PLATE_AREA = 30000
    MIN_ASPECT_RATIO = 2.0
    MAX_ASPECT_RATIO = 4.0 