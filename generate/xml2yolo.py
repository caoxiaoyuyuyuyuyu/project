import os
import xml.etree.ElementTree as ET


def xml_to_yolo(xml_path, classes):
    """
    将XML标注文件转换为YOLO格式
    :param xml_path: XML文件路径
    :param classes: 类别名称列表（如["MQ"]）
    :return: YOLO格式字符串
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图像尺寸
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    yolo_lines = []

    for obj in root.iter('object'):
        # 获取类别ID
        cls_name = obj.find('name').text
        class_id = classes.index(cls_name)

        # 获取边界框坐标
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # 转换为YOLO格式
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        # 保留6位小数
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        yolo_lines.append(line)

    return '\n'.join(yolo_lines)


def batch_convert(xml_dir, output_dir, classes):
    """
    批量转换XML标注文件
    :param xml_dir: XML文件夹路径
    :param output_dir: 输出文件夹路径
    :param classes: 类别列表
    """
    os.makedirs(output_dir, exist_ok=True)

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        yolo_str = xml_to_yolo(xml_path, classes)

        # 生成对应的txt文件名
        base_name = os.path.splitext(xml_file)[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(txt_path, 'w') as f:
            f.write(yolo_str)


if __name__ == "__main__":
    # ======== 使用示例 ========
    # 定义类别（必须与XML中的<name>标签一致）
    CLASSES = ["MQ"]  # 如果多类别，例如：["cat", "dog", "person"]

    # 输入输出路径
    XML_DIR = r"output_xml_dir"  # XML文件夹路径
    OUTPUT_DIR = r"output_yolo_dir"  # 输出文件夹

    # 执行转换
    batch_convert(XML_DIR, OUTPUT_DIR, CLASSES)
    print(f"转换完成，结果已保存到：{OUTPUT_DIR}")
