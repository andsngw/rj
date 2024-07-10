from ultralytics import YOLO
import cv2

model = YOLO("yolov8x.pt")
results = model("ex4.jpg", save=True, save_txt=True, save_conf=True)
boxes = results[0].boxes

path = "ex4.jpg"
img = cv2.imread(path)

for box in boxes:
    startpoint_x = box.data[0, 0]
    startpoint_y = box.data[0, 1]
    endpoint_x = box.data[0, 2]
    endpoint_y = box.data[0, 3]

    # 各線を描画
    cv2.line(img, (int(startpoint_x), int(startpoint_y)), (int(endpoint_x), int(startpoint_y)), (0, 0, 255), thickness=3)  
    cv2.line(img, (int(startpoint_x), int(startpoint_y)), (int(startpoint_x), int(endpoint_y)), (0, 0, 255), thickness=3)  
    cv2.line(img, (int(endpoint_x), int(endpoint_y)), (int(endpoint_x), int(startpoint_y)), (0, 0, 255), thickness=3)  
    cv2.line(img, (int(endpoint_x), int(endpoint_y)), (int(startpoint_x), int(endpoint_y)), (0, 0, 255), thickness=3)  

cv2.imshow('ex4', img)
cv2.waitKey(0)
cv2.destroyAllWindows()