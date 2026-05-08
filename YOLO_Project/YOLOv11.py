from pathlib import Path
from ultralytics import YOLO

# 获取脚本所在目录
base_dir = Path(__file__).parent

yolo = YOLO("yolo11n.pt")

# 使用相对于脚本的路径
image_path = base_dir / "images" / "Dog" / "0.jpg"
results = yolo(str(image_path), save=True)