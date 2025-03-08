# 接受YOLOv11的输出，results = predictor.postprocess(preds, img, orig_imgs)
# orig_img, path, names, boxes
# x1, y1, x2, y2, conf, cls
import glob
import os

import cv2
import keras_ocr
from matplotlib import pyplot as plt

pipeline = keras_ocr.pipeline.Pipeline()

class TransedResult(object):
    def __init__(self, orig_img, path, names, boxes):
        self.orig_img = orig_img
        self.path = path
        self.names = names
        self.boxes = boxes
        self.result = []
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
            prediction = pipeline.recognize([img_cell])
            self.caculate(prediction)
            print(prediction)
            fig, axs = plt.subplots(nrows=len(image_cells), figsize=(20, 20), squeeze=False)  # 关键参数
            for ax, image, predictions in zip(axs.flat, image_cells):
                keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
            plt.show()  # 添加显示命令
    def caculate(self, prediction):
        # 解析出算式识别结果
        for str in  prediction:
            print(str)
        # pass

if __name__ == '__main__':
    jpg_dir = r"/generate/output_jpg_dir"
    yolo_dir = r"/generate/output_yolo_dir"
    files = glob.glob(jpg_dir + "/*.jpg")
    for file in files[0:1]:
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
        transresult = TransedResult(img_ori, file, name, boxes)
        transresult.trans_box()
































