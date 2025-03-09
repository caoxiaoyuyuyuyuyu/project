import os
import xml.etree.ElementTree as ET


def json_to_xml(json_data,output_name):
    # 创建XML根节点
    root = ET.Element("annotation")

    # 添加基础信息
    ET.SubElement(root, "folder").text = "weight"
    ET.SubElement(root, "filename").text = f"{output_name}.jpg"
    ET.SubElement(root, "path").text = os.path.join("G:\AI\crnn.pytorch-master\generate\ooutput_jpg_dir", f"{output_name}.jpg")

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "900"
    ET.SubElement(size, "height").text = "800"
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(root, "segmented").text = "0"

    # 添加所有算式标注
    for item in json_data:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "MQ"
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(item["coordinates"][0])
        ET.SubElement(bbox, "ymin").text = str(item["coordinates"][1])
        ET.SubElement(bbox, "xmax").text = str(item["coordinates"][2])
        ET.SubElement(bbox, "ymax").text = str(item["coordinates"][3])

    return ET.tostring(root, encoding="unicode")
