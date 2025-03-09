import cv2
import os
from ocr.transLabel import TransedResult

def format_result(result):
    if result is None:
        return "None"
    elif isinstance(result, float):
        return "{:.2f}".format(result)
    return str(result)

def add_ocr_results_to_image(img,image_path,name,boxes):
    results = TransedResult(img, image_path, name, boxes).get_result()
    height, width = img.shape[:2]

    for (i, box) in enumerate(list(boxes)):
        x1, y1, x2, y2 = map(int, box[:4])  # 确保前4个坐标值为整数

        result = results[i]
        formatted_result = format_result(result)

        # 计算检测框的高度
        box_height = y2 - y1

        # 初始字体大小
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 0)  # 改为黑色
        thickness = 1

        # 目标文字高度
        target_height = box_height * 2 / 3

        loop_count = 0
        # 动态调整字体大小，使文字高度接近框高的三分之二
        while True:
            (text_width, text_height), _ = cv2.getTextSize(formatted_result, font, font_scale, thickness)
            if abs(text_height - target_height) < target_height * 0.05:
                break
            if text_height > target_height:
                font_scale -= 0.01
            else:
                font_scale += 0.01
            loop_count += 1
            if loop_count > 100:
                break
            if font_scale < 0.1:
                font_scale = 0.1
                break

        # print(f"Final font scale: {font_scale}, Text height: {text_height}")

        # 计算文字的宽度
        text_width = int(text_height * 2 / 3)

        # 计算文字的位置，放在框的右外侧
        text_x = x2 + 10
        text_y = int(y1 + (y2 - y1) // 2 + text_height // 2)

        # 检查文字位置是否超出图像边界
        if text_x + text_width > width:
            text_x = width - text_width - 10
        if text_y > height:
            text_y = height - 10

        text_x = max(0, text_x)
        text_y = max(0, text_y)

        # 在图像上添加文字
        cv2.putText(img, formatted_result, (text_x, text_y), font, font_scale, color, thickness)

    return img