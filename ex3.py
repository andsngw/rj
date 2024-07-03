import cv2
from ultralytics import YOLO
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

# YOLOモデルの読み込み
model = YOLO("yolov8x-pose.pt")

# 基準画像からベースポイントを取得
results = model("ex1.jpg", save=False, save_txt=False, save_conf=False)
basepoints = results[0].keypoints.data.cpu().numpy()[:, :, :2]  # Keypointsオブジェクトからデータを抽出してCPUに移動
print(basepoints)

# 動画の読み込み
movie = "ex3a.mp4"
cap = cv2.VideoCapture(movie)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # フレームの幅
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # フレームの高さ
fps = float(cap.get(cv2.CAP_PROP_FPS))  # FPS

# スケルトンの接続点
skeleton = [[6, 5], [6, 8], [8, 10], [5, 7], [7, 9], [6, 12], [5, 11], [12, 11], [12, 14], [14, 16], [11, 13], [13, 15]]

i = 1

def process_frame(img, basepoints):
    results = model(img, save=False, save_txt=False, save_conf=False)
    points = results[0].keypoints.data.cpu().numpy()[:, :, :2]  # Keypointsオブジェクトからデータを抽出してCPUに移動

    # 基準点との距離計算
    dif = basepoints - points
    distances = np.linalg.norm(dif, axis=2)
    mean_distance = np.mean(distances)

    threshold = 10  # 距離の閾値を設定

    color = (255, 0, 0)  # 青色っで骨格を表示
    if mean_distance < threshold:
        color = (0, 0, 255)  # マッチするときは赤色に変更

    for start_point, end_point in skeleton:
        cv2.line(img, 
                 (int(points[0][start_point][0]), int(points[0][start_point][1])),
                 (int(points[0][end_point][0]), int(points[0][end_point][1])), 
                 color, thickness=4)
    
    # 丸を描画
    for j in range(5, 17):
        cv2.circle(img, 
                   (int(points[0][j][0]), int(points[0][j][1])), 
                   5, color, thickness=-1)
    
    return img

with ThreadPoolExecutor(max_workers=2) as executor:
    while True:
        # フレーム情報取得
        ret, img = cap.read()

        # 動画が終われば処理終了
        if not ret:
            break

        start_time = time.time()

        future = executor.submit(process_frame, img, basepoints)

        # 処理済みフレームを取得
        img = future.result()

        # 動画表示
        cv2.imshow('Video', img)
        
        elapsed_time = time.time() - start_time
        wait_time = max(1, int(1000 / fps) - int(elapsed_time * 1000))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

        print("Frame: " + str(i))
        i += 1

cap.release()
cv2.destroyAllWindows()