import os

import cv2

from yolo.ultralytics import YOLO
from yolo.addresult import add_ocr_results_to_image

def detect(source):

    model = YOLO(model=r'yolo\weight\best.pt')

    # source: 要预测的图片路径
    results = model.predict(source=source,#图片或视频的源目录
                  conf=0.3,#用于检测的 对象置信阈值，只有置信度高于此阈值的对象才会被检测出来
                  iou=0.5,#非极大值抑制(NMS)的交并比(loU)值
                  imgsz=512,#输入图像尺寸
                  half=False,#使用半精度(FP16)
                  device='cpu',#运行设备，如device=0或device = cpu
                  save=False,#是否保存预测的图像和视频
                  # save_conf=True,#是否将检测结果与置信度分数一起保存
                  # save_crop=True,#是否保存裁剪的图像与结果
                  # show_labels=True,#是否显示预测标签
                  # show_conf=True,#是否显示预测置信度
                  # show_boxes=True,#是否显示预测边界框
                  # line_width=1,#边界框的线宽(如果为 None ，则缩放为图像大小)
                  )

    file_name = os.path.basename(source)
    processed_img = add_ocr_results_to_image(results[0].orig_img, source, file_name, results[0].boxes.data)
    cv2.imshow('img', processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # for r in results:
    #     print(r.boxes.data)
    # ori_img = results[0].orig_img
    # cv2.imshow('img', ori_img)
    # cv2.waitKey(0)
    # print(len(results[0].boxes))
    # print(results[0].boxes.data)
if __name__ == '__main__':
    detect(r'generate/testjpg/math_question0.jpg')