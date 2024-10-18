import math
import env
from .point import Point
from .yolov11 import yolov11

class BoundingBox(object):

    def __init__(self, _class_name, _x1, _y1, _x2, _y2):
        self.class_name = _class_name
        self.top_left = Point(x=_x1, y=_y1)
        self.bottom_right = Point(x=_x2, y=_y2)
        
        self.center = self.get_center()
        self.distance = self.calculate_distance_from_camera()

    def get_center(self):
        center_x = (self.top_left.x + self.bottom_right.x) / 2
        center_y = (self.top_left.y + self.bottom_right.y) / 2
        return (center_x, center_y)

    def calculate_distance_from_camera(self):
        real_object_height = yolov11.object_height.get(self.class_name)
        pixel_height = abs(self.bottom_right.y - self.top_left.y)

        # Avoid division by zero
        if pixel_height == 0:
            return float('inf')

        distance = (real_object_height * env.DEFAULT_FOCAL_LENGTH) / pixel_height
        return distance

    def is_safe_distance(self):
        return self.distance > env.SAFE_DISTANCE_THRESHOLD

    def calculate_real_distance_to(self, other_bbox):
        """ Calculate the real-world distance to another bounding box """
        # Calculate the pixel distance between centers
        pixel_distance = math.sqrt(
            (self.center[0] - other_bbox.center[0]) ** 2 +
            (self.center[1] - other_bbox.center[1]) ** 2
        )

        # Calculate the average depth of the two objects
        average_depth = (self.distance + other_bbox.distance) / 2

        # Convert pixel distance to real-world distance using average depth
        real_distance = (pixel_distance * average_depth) / env.DEFAULT_FOCAL_LENGTH
        return real_distance

    def draw(self, cv2, image):
        # Set bounding box color based on distance safety
        bounding_color = env.SAFE_COLOR if self.is_safe_distance() else env.UNSAFE_COLOR

        # Draw the bounding box
        cv2.rectangle(
            img=image,
            pt1=(self.top_left.x, self.top_left.y),
            pt2=(self.bottom_right.x, self.bottom_right.y),
            color=bounding_color,
            thickness=3
        )
        
        # Annotate the class name and distance
        cv2.putText(
            img=image,
            text=f"{self.class_name} - D:{self.distance:.2f}m",
            org=(self.top_left.x, self.top_left.y - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.9,
            color=bounding_color,
            thickness=2
        )
