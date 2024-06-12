import cv2
from ultralytics import YOLO
import numpy as np

# YOLOモデルの読み込み
model = YOLO("yolov8x-pose.pt")

# 画像を読み込む
images = ["ex1.jpg", "ex2_307.jpg", "ex2_336.jpg", "ex2_2015.jpg", "ex2_3077.jpg", "ex2_5175.jpg"] 
keypoints_list = {} #それぞれの画像におけるx,y座標のリスト
sum_distances = {}  # 画像間の距離を格納する辞書

for image in images:
    results = model(image, save=True, save_txt=True, save_conf=True)
    keypoints_list[image] = results[0].keypoints.data[:, :, :2]

    #画像の読み込み
    path = image
    img = cv2.imread(path)


# 画像ごとにキーポイントの座標の差分を計算
base_image = images[0]

for image in images:
    dif = keypoints_list[base_image] - keypoints_list[image]
    dis = np.linalg.norm(dif)
    sum_distances[image] = dis

# 距離が小さい順に画像を並び替えてプリント
sorted_images = sorted(sum_distances, key=sum_distances.get)
for image in sorted_images:
    print(f"Image: {image}")