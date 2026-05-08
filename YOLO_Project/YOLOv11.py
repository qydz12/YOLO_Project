from ultralytics import YOLO
yolo = YOLO("yolo11n.pt")
results = yolo("YOLO_Project/images/example/", save=True, name="my_results")
print("识别完成")