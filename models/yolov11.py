from ultralytics import YOLO

class YoloV11(YOLO):
  target_classes = {
    0: "person",           # Pedestrians
    1: "bicycle",          # Bicycles
    2: "car",              # Cars
    3: "motorcycle",       # Motorcycles
    5: "bus",              # Buses
    7: "truck",            # Trucks
    9: "traffic light",    # Traffic lights (optional)
    11: "stop sign",       # Stop signs (optional)
    12: "parking meter"
  }
  
  def __init__(self, model="./data/weights/yolov11n.pt", task=None, verbose=False):
    super().__init__(model, task, verbose)

yolov11 = YoloV11()