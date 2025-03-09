import os
import random
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

from generate.json2xml import json_to_xml

# 配置参数
config = {
    "image_size": (900, 800),
    "font_paths": [
        os.path.join(r"G:\AI\crnn.pytorch-master\fonts", f)
        for f in os.listdir(r"G:\AI\crnn.pytorch-master\fonts")
        if f.lower().endswith(('.ttf', '.otf', '.ttc'))
    ],
    "font_size_range": (24, 26),
    "num_expressions": 52,
    "output_xml_dir": "output_xml_dir",
    "output_jpg_dir": "testjpg",
    "padding": 40,
    "noise_level": 0.01,
    "texture_alpha": 0.03,
    "text_color": (0, 0, 0),
    "background_color": (255, 255, 255)
}


def add_texture(image):
    """添加极细纹理"""
    texture = Image.new('L', image.size, 255)
    draw = ImageDraw.Draw(texture)
    for i in range(0, image.width, 50):
        draw.line([(i, 0), (i, image.height)], fill=230, width=1)
    for j in range(0, image.height, 50):
        draw.line([(0, j), (image.width, j)], fill=230, width=1)

    return Image.composite(
        image,
        Image.new('RGB', image.size, config["background_color"]),
        texture.point(lambda x: x * config["texture_alpha"])
    )


def generate_expression():
    """生成随机算术表达式（修复运算符）"""
    operators = ['+', '-', '×', '÷']  # 修正运算符列表
    op = random.choice(operators)
    a = random.randint(1, 99)
    b = random.randint(1, 99)

    if op == '÷':
        b = random.randint(1, 20)
        a = b * random.randint(1, 20)
    elif op == '-':
        a, b = max(a, b), min(a, b)

    return f"{a} {op} {b} ="


def add_noise(image):
    """添加随机噪点"""
    pixels = image.load()
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            if random.random() < config["noise_level"]:
                noise = random.randint(50, 200)
                pixels[i, j] = (noise, noise, noise)


def generate_image(output_name):
    img = Image.new('RGB', config["image_size"], config["background_color"])
    draw = ImageDraw.Draw(img)

    cols = 4
    col_width = config["image_size"][0] // cols
    # x_start = config["padding"]
    x_start = 0
    y = config["padding"]
    max_height = 0
    expressions_data = []

    current_col = 0
    x = x_start

    for _ in range(config["num_expressions"]):
        # 生成文本
        text = generate_expression()
        font_size = random.randint(*config["font_size_range"])
        font = ImageFont.truetype(random.choice(config["font_paths"]), font_size)

        # 计算列位置
        x = x_start + current_col * col_width
        x_center = x + (col_width - font.getbbox(text)[2]) // 2  # 水平居中

        # 绘制文本并获取实际坐标
        draw.text((x_center, y), text, fill=config["text_color"], font=font)
        bbox = draw.textbbox((x_center, y), text, font=font)

        # 记录实际坐标
        expressions_data.append({
            "text": text,
            "coordinates": [
                max(0, bbox[0]),  # xmin
                max(0, bbox[1]),  # ymin
                min(config["image_size"][0], bbox[2]),  # xmax
                min(config["image_size"][1], bbox[3])  # ymax
            ]
        })

        # 更新布局参数
        current_col += 1
        text_height = bbox[3] - bbox[1]
        max_height = max(max_height, text_height)

        # 换行处理
        if current_col >= cols:
            current_col = 0
            y += max_height + config["padding"]
            max_height = 0

    # 保存结果
    img.save(os.path.join(config["output_jpg_dir"], f"{output_name}.jpg"))
    # with open(os.path.join(config["output_xml_dir"], f"{output_name}.xml"), "w") as f:
    #     f.write(json_to_xml(expressions_data, output_name))


if __name__ == "__main__":
    os.makedirs(config["output_jpg_dir"], exist_ok=True)
    os.makedirs(config["output_xml_dir"], exist_ok=True)
    for i in range(2):
        generate_image(f"math_question{i}")
