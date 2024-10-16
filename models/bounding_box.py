import env
from .point import Point
from .yolov8 import yolov8

class BoundingBox(object):

  def __init__(self, _class_name, _x1, _y1, _x2, _y2):
    self.class_name = _class_name
    self.top_left = Point(x=_x1, y=_y1)
    self.bottom_right = Point(x=_x2, y=_y2)
    
    self.center = self.get_center()
    self.distance = self.calculate_distance_from_camera()

  def get_center(self):
    center_x = (self.top_left.x + self.bottom_right.x)/2
    center_y = (self.top_left.y + self.bottom_right.y)/2
    return (center_x, center_y)

  def calculate_distance_from_camera(self):
    real_object_width = yolov8.object_height.get(self.class_name)
    pixel_width = abs(self.bottom_right.y - self.top_left.y)

    distance = (real_object_width * env.DEFAULT_FOCAL_LENGTH) / pixel_width
    return distance

  def is_safe_distance(self):
    return self.distance > env.SAFE_DISTANCE_THRESHOLD
  
  def draw(self, cv2, image):
    if self.is_safe_distance():
        bounding_color = env.SAFE_COLOR
    else:
        bounding_color = env.UNSAFE_COLOR

    cv2.rectangle(
        img=image,
        pt1=(self.top_left.x, self.top_left.y),
        pt2=(self.bottom_right.x, self.bottom_right.y),
        color=bounding_color,
        thickness=3
    )
        
    cv2.putText(
        img=image,
        text=f"{self.class_name} - D:{self.distance:.2f}m",
        org=(self.top_left.x, self.top_left.y - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.9,
        color=bounding_color,
        thickness=2
    )
