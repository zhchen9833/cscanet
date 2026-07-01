import cv2


def resize_image(image=None, scale_factor=None, new_height=None, new_width=None):
    if scale_factor is not None:
        assert new_width is None and new_height is None
    if scale_factor is None:
        assert new_width is not None and new_height is not None

    # 读取图像
    if isinstance(image, str):
        image = cv2.imread(image)
    else:
        image = image

    if image is None:
        print("Error: Image not found.")
        return None

    # 获取原图像的尺寸
    original_height, original_width = image.shape[:2]

    # 计算新的尺寸
    if scale_factor is not None:
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
    else:
        new_width = new_width
        new_height = new_height

    # 使用OpenCV的resize函数来缩放图像
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image
