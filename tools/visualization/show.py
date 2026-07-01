import glob
import os

import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt


def TwoPercentLinear(image, max_out=255, min_out=0):  # 2%的线性拉伸
    b, g, r = cv2.split(image)  # 分开三个波段

    def gray_process(gray, maxout=max_out, minout=min_out):
        high_value = np.percentile(gray, 98)  # 取得98%直方图处对应灰度
        low_value = np.percentile(gray, 2)  # 同理
        truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
        processed_gray = ((truncated_gray - low_value) / (high_value - low_value)) * (maxout - minout)  # 线性拉伸嘛
        return processed_gray

    r_p = gray_process(r)
    g_p = gray_process(g)
    b_p = gray_process(b)
    result = cv2.merge((b_p, g_p, r_p))  # 合并处理后的三个波段
    return np.uint8(result)


def crop_and_square(img, crop_coords):
    """
    精简版图像裁剪拉伸函数

    参数:
        image: 输入图像(路径或HWC格式数组)
        crop_coords: 裁剪区域(x1,y1,x2,y2)

    返回:
        处理后的正方形图像
    """
    # 读取图像（支持路径或数组输入）
    # img = cv2.imread(image) if isinstance(image, str) else image
    # if len(img.shape) == 3:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 执行裁剪
    x1, y1, x2, y2 = crop_coords
    cropped = img[y1:y2, x1:x2]

    # 计算拉伸比例
    output_size = x2 - x1
    new_size = (output_size, output_size)

    # 等比缩放
    resized = cv2.resize(cropped, new_size, interpolation=cv2.INTER_LINEAR)

    # 创建空白画布并居中放置
    result = np.zeros((output_size, output_size, 3), dtype=resized.dtype)
    pad_x = (output_size - new_size[0]) // 2
    pad_y = (output_size - new_size[1]) // 2
    result[pad_y:pad_y + new_size[1], pad_x:pad_x + new_size[0]] = resized

    return result


def replace_with_zoom(image, rect_coords, zoom_scale=2.0):
    """
    将局部放大区域直接替换到原图右下角

    参数:
        image: 输入图像 (HWC格式)
        rect_coords: 矩形区域坐标 (x1,y1,x2,y2)
        zoom_scale: 放大倍数 (默认2.0)
        display: 是否显示结果 (默认True)

    返回:
        处理后的图像 (右下角被替换为放大区域)
    """
    # 深拷贝避免修改原图
    img = image.copy()
    x1, y1, x2, y2 = rect_coords

    # 1. 裁剪并放大目标区域
    cropped = img[y1:y2, x1:x2]
    h, w = cropped.shape[:2]
    zoomed = cv2.resize(cropped, (int(w * zoom_scale), int(h * zoom_scale)),
                        interpolation=cv2.INTER_CUBIC)

    # 为放大区域添加红色边框
    cv2.rectangle(zoomed, (0, 0), (zoomed.shape[1] - 1, zoomed.shape[0] - 1),
                  (255, 0, 0), 1)

    # 2. 计算右下角替换区域尺寸
    zh, zw = zoomed.shape[:2]
    img_h, img_w = img.shape[:2]

    # 确保替换区域不超出图像边界
    replace_w = min(zw, img_w)
    replace_h = min(zh, img_h)

    # 3. 直接替换右下角像素
    img[img_h - replace_h:, img_w - replace_w:] = zoomed[:replace_h, :replace_w]

    # 4. 绘制红色矩形标记原区域
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    return img


# 使用示例
if __name__ == "__main__":
    # show for HD 1-15.tif wo rectangle
    # img_path = 'samples/1_15-D3-HD-e30.png'
    # test_img = cv2.imread(img_path)
    # result = crop_and_square(test_img, crop_coords=(140, 0, 512, 507))
    #
    # imageio.imwrite('results/1_15-D3-HD-e30-wo-rectangle.png', result)

    # img_path = 'samples/2_12_1_3-D3-AVIRIS-e30.png'
    # test_img = cv2.imread(img_path)
    # test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    #
    # square = 45
    # start_x = 38
    # start_y = 166
    # show = replace_with_zoom(test_img, rect_coords=(start_x, start_y, start_x + square, start_y + square),
    #                          zoom_scale=2.0)
    # imageio.imwrite(img_path.replace('samples', 'results'), show)

    # ------------------------------------------------ HD ------------------------------------------------ #
    # show for HD 1-15.tif
    for filename in os.listdir('HD/1-15/samples'):
        img_path = os.path.join('HD/1-15/samples', filename)
        test_img = imageio.v2.imread(img_path)
        # test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_img = TwoPercentLinear(test_img)
        result = crop_and_square(test_img, crop_coords=(140, 0, 512, 507))

        square = 62
        start_x = 285
        start_y = 11
        show = replace_with_zoom(result, rect_coords=(start_x, start_y, start_x + square, start_y + square),
                                 zoom_scale=2.0)
        imageio.imwrite(img_path.replace('samples', 'results'), show)

    # params for UAV
    # square = 44
    # start_x = 5
    # start_y = 97

    # params for AVIRIS
    # square = 45
    # start_x = 38
    # start_y = 166

    # ------------------------------------------------ HDD ------------------------------------------------ #
    # show for HDD
    for filename in os.listdir('HDD/GF9/samples'):
        img_path = os.path.join('HDD/GF9/samples', filename)
        test_img = imageio.v2.imread(img_path)
        test_img = TwoPercentLinear(test_img)
        # cv2.rectangle(test_img, (240, 130), (512, 512), (255, 0, 0), 1)

        # 裁剪并拉伸
        result = crop_and_square(test_img, crop_coords=(245, 145, 512, 507))

        square = 40
        start_x = 125
        start_y = 105
        show = replace_with_zoom(result, rect_coords=(start_x, start_y, start_x + square, start_y + square),
                                 zoom_scale=2.0)
        imageio.imwrite(img_path.replace('samples', 'results'), show)

    # params for HDD
    # crop_coords: (245, 145, 512, 507) zoom: square = 40 start_x = 125 start_y = 105

    # 使用Matplotlib展示
    # plt.figure(figsize=(15, 5))
    #
    # plt.subplot(1, 3, 1)
    # plt.imshow(test_img)
    # plt.title("Original Image")
    # plt.axis('off')
    #
    # plt.subplot(1, 3, 2)
    # plt.imshow(result)
    # plt.title("Cropped & Squared")
    # plt.axis('off')
    #
    # plt.subplot(1, 3, 3)
    # plt.imshow(show)
    # plt.title("Show")
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.savefig('compare_fake.png')
