import cv2

for i in range(10):

    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if cap.isOpened():
        print(f"摄像头 {i} 可用")
    else:
        print(f"摄像头 {i} 不可用")

    cap.release()