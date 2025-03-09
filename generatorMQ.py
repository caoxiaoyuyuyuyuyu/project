import os
import random
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

def random_color(lower_val, upper_val):
    return [random.randint(lower_val, upper_val),
            random.randint(lower_val, upper_val),
            random.randint(lower_val, upper_val)]


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


class GeneratorMQ(Dataset):
    def __init__(self, alpha='0123456789+-×÷='):
        super(GeneratorMQ, self).__init__()
        self.dataset_size = 1000
        self.alpha = alpha
        self.alpha_list = list(alpha)
        self.im_h = 32
        self.im_w = 138
        self.min_len = 5
        self.max_len_list = [16, 19, 24, 26]
        self.max_len = max(self.max_len_list)
        self.font_size_list = [30, 25, 20, 18]
        self.font_list = []  # 二维列表[size,font]
        self.font_paths = [
            os.path.join(r"fonts", f)
            for f in os.listdir(r"fonts")
            if f.lower().endswith(('.ttf', '.otf', '.ttc'))
        ]
        for size in self.font_size_list:
            self.font_list.append([ImageFont.truetype(font_path, size=size)
                                   for font_path in self.font_paths])

        self.alpha_set = set(alpha)

    def __len__(self):
        return self.dataset_size  # 返回数据集的总样本数

    def gen_background(self):
        """
        生成背景;随机背景|纯色背景|合成背景
        :return:
        """
        a = random.random()
        pure_bg = np.ones((self.im_h, self.im_w, 3)) * np.array(random_color(0, 100))
        random_bg = np.random.rand(self.im_h, self.im_w, 3) * 100
        if a < 0.1:
            return random_bg
        elif a < 0.8:
            return pure_bg
        else:
            b = random.random()
            mix_bg = b * pure_bg + (1 - b) * random_bg
            return mix_bg

    def draw_text(self, draw, text, font, color, char_w, char_h):
        """
        水平方向文字合成
        :param draw:
        :param text:
        :param font:
        :param color:
        :param char_w:
        :param char_h:
        :return:
        """
        text_w = len(text) * char_w
        h_margin = max(self.im_h - char_h, 1)
        w_margin = max(self.im_w - text_w, 1)
        x_shift = np.random.randint(0, w_margin)
        y_shift = np.random.randint(0, h_margin)
        i = 0
        while i < len(text):
            draw.text((x_shift, y_shift), text[i], color, font=font)
            i += 1
            x_shift += char_w
            y_shift = np.random.randint(0, h_margin)
            # 如果下个字符超出图像，则退出
            if x_shift + char_w > self.im_w:
                break
        return text[:i]

    def gen_image(self):
        idx = np.random.randint(len(self.max_len_list))
        image = self.gen_background()
        image = image.astype(np.uint8)
        # 随机选择size,font
        size_idx = np.random.randint(len(self.font_size_list))
        font = ImageFont.truetype(random.choice(self.font_paths), self.font_size_list[size_idx])

        # 使用generate_expression生成算术表达式
        text = generate_expression()

        # 计算字体的w和h
        bbox = font.getbbox(text)
        w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]
        char_w = int(w / len(text))

        # 写文字，生成图像
        im = Image.fromarray(image)
        draw = ImageDraw.Draw(im)
        color = tuple(random_color(105, 255))
        text = self.draw_text(draw, text, font, color, char_w, char_h)
        target_len = len(text)  # target_len可能变小了

        indices = []
        for c in text:
            try:
                indices.append(self.alpha.index(c))
            except ValueError:
                continue  # 或处理为特殊标记
        if not indices:  # 无有效字符时重新生成
            return self.gen_image()
        indices = np.array(indices)
        target_len = len(indices)

        image = np.array(im)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if random.random() > 0.5:
            image = 255 - image
        return image, indices, target_len

    def __getitem__(self, item):
        image, indices, target_len = self.gen_image()

        # 图像预处理
        image = np.transpose(image[:, :, np.newaxis], axes=(2, 1, 0))  # [H,W,C]=>[C,W,H]
        image = image.astype(np.float32) / 255.
        image -= 0.5
        image /= 0.5

        # 构建目标标签
        target = np.zeros(shape=(self.max_len,), dtype=np.int64)
        target[:target_len] = indices
        input_len = self.im_w // 4 - 3  # 根据CRNN模型要求计算

        return (
            torch.from_numpy(image),
            torch.from_numpy(target),
            torch.tensor([input_len], dtype=torch.int32),
            torch.tensor([target_len], dtype=torch.int32)
        )


if __name__ == '__main__':
    dataset = GeneratorMQ()
    image, target, input_len, target_len = dataset[0]

    # 转换为numpy并处理维度
    image = image.numpy().squeeze(0)  # [C,H,W]=[1,32,138] → [32,138]

    # 反归一化
    image = (image * 0.5 + 0.5) * 255
    image = image.astype(np.uint8)

    # 显示时转置宽高
    cv2.imshow('image', cv2.transpose(image))  # [32,138] → [138,32]
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 15 + 56 = 20
