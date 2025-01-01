import numpy as np
from scipy.ndimage import morphology


def my_erode(iml, b):
    # 将输入的图像和结构元素转换为numpy数组
    iml = np.array(iml, dtype=bool)
    b = np.array(b, dtype=bool)

    # 执行腐蚀操作
    r = morphology.binary_erosion(iml, structure=b)

    # 将结果转换回图像格式
    r = np.array(r, dtype=np.uint8)

    return r


# 定义原始图像和结构元素
original_image = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0]
]

structuring_element = [
    [1, 1],
    [1, 0]
]

# 调用函数并打印结果
eroded_image = my_erode(original_image, structuring_element)
print(eroded_image)