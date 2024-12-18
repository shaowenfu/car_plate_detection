import cv2
from ..config.config import Config

def resize_image(image, width=None, height=None):
    """
    调整图像大小，保持宽高比
    """
    if width is None and height is None:
        width = Config.RESIZE_WIDTH
        height = Config.RESIZE_HEIGHT
        
    dim = None
    (h, w) = image.shape[:2]
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def standardize_size(image):
    """
    将图像标准化为固定尺寸
    """
    return cv2.resize(image, (Config.RESIZE_WIDTH, Config.RESIZE_HEIGHT))
