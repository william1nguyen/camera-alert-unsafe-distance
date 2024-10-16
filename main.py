import cv2
import argparse
from errors import OpenCameraException
from models.yolov8 import yolov8
from env import *
from models.bounding_box import BoundingBox

object_detech_model = yolov8

def predict(frame):
    results = object_detech_model.track(frame, stream=True)
    return results

def show_webcam_results(image, predicted_results, target_classes=[]):
    bounding_boxes = []
    for r in predicted_results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls in target_classes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                bounding_box = BoundingBox(cls, x1, y1, x2, y2)
                bounding_boxes.append(bounding_boxes)
                bounding_box.draw(cv2=cv2, image=image)

    cv2.imshow('Webcam', image)

def start_webcam(cap):
    if not cap.isOpened():
        raise OpenCameraException()
    while True:
        ret, frame = cap.read()
        if ret:
            predicted_results = predict(frame=frame)
            target_classes = list(object_detech_model.target_classes.keys())
            show_webcam_results(
                image=frame, 
                predicted_results=predicted_results,
                target_classes=target_classes,
            )
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def release(cap):
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', dest='test', type=str, help='test destination')
    args = parser.parse_args()

    if args.test:
        cap = cv2.VideoCapture(args.test)
    else: 
        cap = cv2.VideoCapture(0)
    start_webcam(cap=cap)
    release(cap=cap)