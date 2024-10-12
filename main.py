import cv2
import argparse
import numpy as np
from errors import OpenCameraException
from models.yolov8 import yolov8
from env import *

object_detech_model = yolov8

def predict(frame):
    results = object_detech_model.track(frame, stream=True)
    return results

def calculate_distance(focal_length, real_object_width, pixel_width):
    distance = (real_object_width * focal_length) / pixel_width
    return distance

def check_safe_distance(distance):
    return distance > SAFE_DISTANCE_THRESHOLD

def show_webcam_results(image, predicted_results, target_classes=[]):
    for r in predicted_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            confidence = box.conf[0]

            if cls in target_classes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                distance = calculate_distance(
                    DEFAULT_FOCAL_LENGTH,
                    object_detech_model.object_height.get(cls),
                    abs(x2 - x1)
                )

                is_safe = check_safe_distance(distance=distance)
                if is_safe:
                    bounding_color = SAFE_COLOR
                else:
                    bounding_color = UNSAFE_COLOR

                cv2.rectangle(
                    img=image,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=bounding_color,
                    thickness=3
                )
                    
                # Display depth on the image
                cv2.putText(
                    img=image,
                    text=f"{object_detech_model.target_classes[cls]} - D:{distance:.2f}m",
                    org=(x1, y1 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.9,
                    color=bounding_color,
                    thickness=2
                )

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