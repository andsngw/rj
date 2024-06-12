import cv2
from ultralytics import YOLO
import numpy as np

# YOLOモデルの読み込み
model = YOLO("yolov8x-pose.pt")



# 画像を読み込む
images = ["ex1.jpg", "ex2_307.jpg", "ex2_336.jpg", "ex2_2015.jpg", "ex2_3077.jpg", "ex2_5175.jpg"] 
keypoints_list = {} #それぞれの画像におけるx,y座標のリスト
sum = {}


for image in images:
    results = model(image, save=True, save_txt=True, save_conf=True)
    keypoints_list[image] = results[0].keypoints.data[:, :, :2]

    #画像の読み込み
    path = image
    img = cv2.imread(path)

# 画像ごとにキーポイントの座標の差分を計算
sum_dif_x = {}
sum_dif_y = {}
base_image = images[0]

for image in images:
    dif = keypoints_list[base_image] - keypoints_list[image]
    sum_dif_x[image] = abs(dif[:, 0]).sum()
    sum_dif_y[image] = abs(dif[:, 1]).sum()

    sum[image] = pow((pow(sum_dif_x[image], 2) + pow(sum_dif_y[image], 2)), 0.5)

sorted_images = sorted(sum, key=sum.get)
for image in sorted_images:
    print(f"Image: {image}")