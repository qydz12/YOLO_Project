from ultralytics import YOLO
import cv2
import time
import torch
import GPUtil
import psutil

# =========================
# 1. 检测设备
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("当前设备:", device)

# =========================
# 2. 加载模型
# =========================
model = YOLO("yolo11n.pt")
model.to(device)

# =========================
# 3. 打开摄像头
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # =========================
    # 4. 推理开始时间
    # =========================
    start_time = time.time()

    # YOLO推理
    results = model(
        frame,
        imgsz=640,
        conf=0.5,
        verbose=False
    )

    # 推理结束时间
    end_time = time.time()

    # 推理耗时(ms)
    inference_time = (end_time - start_time) * 1000

    # =========================
    # 5. 绘制结果
    # =========================
    annotated_frame = results[0].plot()

    # =========================
    # 6. FPS
    # =========================
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # =========================
    # 7. GPU信息
    # =========================
    gpus = GPUtil.getGPUs()

    if len(gpus) > 0:
        gpu = gpus[0]

        gpu_load = gpu.load * 100
        gpu_memory = gpu.memoryUsed
        gpu_memory_total = gpu.memoryTotal
        gpu_temp = gpu.temperature

    else:
        gpu_load = 0
        gpu_memory = 0
        gpu_memory_total = 0
        gpu_temp = 0

    # =========================
    # 8. CPU信息
    # =========================
    cpu_usage = psutil.cpu_percent()

    # =========================
    # 9. 显示信息
    # =========================
    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"GPU Load: {gpu_load:.1f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"GPU Memory: {gpu_memory:.0f}/{gpu_memory_total:.0f} MB",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"GPU Temp: {gpu_temp:.1f} C",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"CPU Usage: {cpu_usage:.1f}%",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Inference: {inference_time:.1f} ms",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # =========================
    # 10. 显示画面
    # =========================
    cv2.imshow("YOLOv11 GPU Monitor", annotated_frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()