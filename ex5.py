import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

video_path = "ex5.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame, save=True, save_txt=True, save_conf=True)
        boxes = results[0].boxes


        for box in boxes:
            startpoint_x = box.data[0, 0]
            startpoint_y = box.data[0, 1]
            endpoint_x = box.data[0, 2]
            endpoint_y = box.data[0, 3]

            cv2.line(frame, (int(startpoint_x), int(startpoint_y)), (int(endpoint_x), int(startpoint_y)), (0, 0, 255), thickness=3)  
            cv2.line(frame, (int(startpoint_x), int(startpoint_y)), (int(startpoint_x), int(endpoint_y)), (0, 0, 255), thickness=3)  
            cv2.line(frame, (int(endpoint_x), int(endpoint_y)), (int(endpoint_x), int(startpoint_y)), (0, 0, 255), thickness=3)  
            cv2.line(frame, (int(endpoint_x), int(endpoint_y)), (int(startpoint_x), int(endpoint_y)), (0, 0, 255), thickness=3)  


            cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()