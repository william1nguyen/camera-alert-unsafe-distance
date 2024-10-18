from ultralytics import YOLO

class Yolov11(YOLO):
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

  object_height = {
    0: 0.5,     # Pedestrians
    1: 0.65,    # Bicycles
    2: 1.9,    # Cars
    3: 0.8,    # Motorcycles
    5: 2.5,    # Buses
    7: 2.55,    # Trucks
    9: 0.4,    # Traffic lights (optional)
    11: 0.75,    # Stop signs (optional)
    12: 0.25,
  }

  def __init__(self, model="./data/weights/yolov11n.pt", task=None, verbose=False):
    super().__init__(model, task, verbose)

yolov11 = Yolov11()