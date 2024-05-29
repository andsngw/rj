import cv2
from ultralytics import YOLO
import numpy as np
#yoloを実行
model = YOLO("yolov8x-pose.pt")
results = model("ex1.jpg", save=True, save_txt=True, save_conf=True)
keypoints = results[0].keypoints
print(keypoints.data)

#画像の読み込み
path = "ex1.jpg"
img = cv2.imread(path)

# #線を描画
skeleton=[[[6,5],[6,8],[8,10],[5,7],[7,9],[6,12],[5,11],[12,11],[12,14],[14,16],[11,13],[13,15]]]
for i in range(0,12):
    start_point=skeleton[0][i][0]
    end_point=skeleton[0][i][1]
    cv2.line(img, (int(keypoints.data[0][start_point][0]), int(keypoints.data[0][start_point][1])),(int(keypoints.data[0][end_point][0]),
                                                     int(keypoints.data[0][end_point][1])), (0, 0, 0),thickness=4)
#丸を描画
for i in range(5,17):
    cv2.circle(img, (int(keypoints.data[0][i][0]),int(keypoints.data[0][i][1])), 5, (255, 0, 0), thickness=-1)
cv2.imshow('ex1', img)
cv2.waitKey(0)
cv2.destroyAllWindows()