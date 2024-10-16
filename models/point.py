import math

class Point(object):
  def __init__(self, x, y):
    self.x = x
    self.y = y
  
  def get_distance(self, other_point):
    distance = math.sqrt(
      abs(self.x - other_point.x) ** 2 + abs(self.y - other_point.y) ** 2
    )

    return distance