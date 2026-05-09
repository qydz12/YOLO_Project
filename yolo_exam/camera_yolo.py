from ultralytics import YOLO
import cv2
import time
import torch

# =========================
# 1. 加载模型（GPU）
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO("yolo11n.pt")
model.to(device)

# =========================
# 2. 打开摄像头
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 设置分辨率（越小越快）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================
# 3. FPS计算
# =========================
prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # =========================
    # 4. YOLO推理（关键优化）
    # =========================
    results = model(
        frame,
        imgsz=640,      # 输入尺寸
        conf=0.5,       # 置信度
        verbose=False   # 关闭日志
    )

    # 绘制结果
    annotated_frame = results[0].plot()

    # =========================
    # 5. FPS计算
    # =========================
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # =========================
    # 6. 显示画面
    # =========================
    cv2.imshow("YOLOv11 High FPS", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()