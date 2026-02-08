from ultralytics import YOLO

yolo = YOLO("yolov8n.pt")

# Automatically load all class names from YOLO model
CLASS_NAMES = yolo.model.names

def detect_objects(frame):
    results = yolo(frame)
    objs = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls)
        conf = float(box.conf)

        width = abs(x2 - x1)
        height = abs(y2 - y1)
        area = width * height

        objs.append({
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "area": area,
            "class": cls,
            "name": CLASS_NAMES[cls],   # auto name
            "conf": conf
        })

    # Sort by size: largest object first
    objs.sort(key=lambda x: x["area"], reverse=True)
    return objs
