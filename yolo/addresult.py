import cv2
import os
from ocr.transLabel import TransedResult


def read_yolo_txt(txt_path, image_width, image_height):
    """
    :param txt_path: TXT 文件路径
    :param image_width: 图像宽度
    :param image_height: 图像高度
    :return: 检测框信息列表
    该函数用于读取 YOLO 模型输出的 TXT 格式检测结果文件，并将文件中存储的归一化坐标转换为实际的像素坐标。
    """
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                x_center = parts[1]
                y_center = parts[2]
                width = parts[3]
                height = parts[4]

                # 转换为实际像素坐标
                x1 = int((x_center - width / 2) * image_width)
                y1 = int((y_center - height / 2) * image_height)
                x2 = int((x_center + width / 2) * image_width)
                y2 = int((y_center + height / 2) * image_height)

                boxes.append({
                    "class_id": class_id,
                    "box": [x1, y1, x2, y2]
                })
    return boxes


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


def add_ocr_results_to_images(image_dir, yolo_dir, output_dir):
    """
    将 OCR 结果添加到图像上并保存
    :param image_dir: 图像文件夹路径
    :param yolo_dir: YOLO 检测结果 TXT 文件文件夹路径
    :param output_dir: 输出结果图像的文件夹路径
    """
    os.makedirs(output_dir, exist_ok=True)
    for image_name in os.listdir(image_dir):
        if not image_name.endswith(('.jpg', '.png', '.jpeg')):
            continue

        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue

        height, width = image.shape[:2]
        txt_name = os.path.splitext(image_name)[0] + '.txt'
        txt_path = os.path.join(yolo_dir, txt_name)
        boxes = read_yolo_txt(txt_path, width, height)

        # 提取检测框的坐标列表
        box_coords = [box_info["box"] for box_info in boxes]

        # 调用 TransedResult 类的 get_result 方法获取 OCR 结果
        current_ocr_results = TransedResult(image, txt_path, image_name, box_coords).get_result()
        print(f"Image: {image_name}, OCR Results: {current_ocr_results}")

        img_info = {
            "image_path": image_path,
            "boxes": []
        }

        for i, box_info in enumerate(boxes):
            result = current_ocr_results[i] if i < len(current_ocr_results) else "No result"
            box_info["result"] = result
            img_info["boxes"].append(box_info)

        for box_info in img_info["boxes"]:
            box = box_info["box"]
            x1, y1, x2, y2 = box
            result = box_info["result"]
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
            text_y = y1 + (y2 - y1) // 2 + text_height // 2

            # 检查文字位置是否超出图像边界
            if text_x + text_width > width:
                text_x = width - text_width - 10
            if text_y > height:
                text_y = height - 10

            # 在图像上添加文字
            cv2.putText(image, formatted_result, (text_x, text_y), font, font_scale, color, thickness)

        # 保存添加文字后的图像
        output_path = os.path.join(output_dir, f"output_{image_name}")
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path}")

        return output_path


if __name__ == "__main__":
    #
    image_dir = r"generate\output_jpg_dir"#实际原图存储路径
    yolo_dir = r"generate\output_yolo_dir"#实际yolo.txt存储路径
    output_dir = r"outimg"#实际输出保存路径

    add_ocr_results_to_images(image_dir, yolo_dir, output_dir)