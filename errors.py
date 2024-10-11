class OpenCameraException(Exception):
  def __init__(self, *args: object) -> None:
    super().__init__(*args)
    self.message = "Could not open webcam"
    self.code = 400

  def __str__(self) -> str:
    return self.message
