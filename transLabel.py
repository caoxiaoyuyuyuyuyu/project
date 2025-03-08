# 接受YOLOv11的输出，results = predictor.postprocess(preds, img, orig_imgs)
# orig_img, path, names, boxes
# x1, y1, x2, y2, conf, cls
import glob
import os
import re

import cv2
import torch
from matplotlib import pyplot as plt

import crnn
from predict import inference_images


class TransedResult(object):
    def __init__(self, orig_img, path, names, boxes):
        self.orig_img = orig_img
        self.path = path
        self.names = names
        self.boxes = boxes
        self.result = []
        self.alpha = '0123456789+-×÷='
        self.weight_path = r"output/crnn.horizontal.101.pth"
        self.net = crnn.CRNN(num_classes=len(self.alpha))
        self.net.load_state_dict(torch.load(self.weight_path, map_location='cpu',
                                       weights_only=False)['model'])
        self.net.eval()
    def __len__(self):
        return len(self.boxes)
    def trans_box(self):
        image_cells = []
        for box in self.boxes:
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]
            img_cell = self.orig_img[int(ymin):int(ymax), int(xmin):int(xmax)]
            image_cells.append(img_cell)
        # ocr识别
        prediction = inference_images(self.net, self.alpha, image_cells)
        for i in range(len(prediction)):
            print(prediction[i])
            # self.result.append(self.caculate(prediction[i]))
        for i in range(len(self.result)):
            print(prediction[i], self.result[i])
        return self.result
    def caculate(self, prediction):
        pattern = r'^\s*\d{1,8}\s*[+\-×÷]\s*\d{1,8}\s*=?$'
        result = None
        if re.match(pattern, prediction):
            prediction = prediction.replace(' ', '')
            prediction = prediction.replace('×', '*')
            prediction = prediction.replace('÷', '/')
            prediction = prediction.replace('=', '')
            # 处理前导零问题
            prediction = re.sub(r'\b0+(\d+)\b', r'\1', prediction)  # 去除数字中的前导零
            # 安全计算
            try:
                result = eval(prediction)
            except:
                result = None
        return result
    def get_result(self):
        if len(self.result) > 0:
            return self.result
        else:
            return self.trans_box()

if __name__ == '__main__':
    jpg_dir = r"generate/output_jpg_dir"

    files = glob.glob(jpg_dir + "/*.jpg")

    for file in files[1:2]:
        name = os.path.basename(file)
        txt_path = os.path.splitext(file)[0] + '.txt'  # 替换扩展名
        txt_path = txt_path.replace('output_jpg_dir', 'output_yolo_dir')  # 替换目录
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                boxes = []
                for line in lines:
                    line = line.strip()
                    line = line.split(' ')
                    # print(line)
                    x_center = int(float(line[1])*900)
                    y_center = int(float(line[2])*800)
                    width = int(float(line[3])*900)
                    height = int(float(line[4])*800)
                    xmin = int(x_center - width/2)
                    ymin = int(y_center - height/2)
                    xmax = int(x_center + width/2)
                    ymax = int(y_center + height/2)
                    boxes.append([xmin, ymin, xmax, ymax])
        img_ori = cv2.imread(file)
        print(TransedResult(img_ori, file, name, boxes).get_result())
































