import cv2
import argparse
import env
from errors import OpenCameraException
from models.yolov11 import yolov11
from env import *
from models.bounding_box import BoundingBox

object_detech_model = yolov11

def predict(frame):
    results = object_detech_model.track(frame, stream=True)
    return results

def show_webcam_results(image, predicted_results, target_classes=[]):
    """ Process and visualize predictions on the webcam feed """
    bounding_boxes = []
    
    for r in predicted_results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls in target_classes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]

                if confidence < env.CONFIDENCE_THRESHOLD: 
                    continue

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                bounding_box = BoundingBox(cls, x1, y1, x2, y2)
                bounding_boxes.append(bounding_box)
                if not bounding_box.is_safe_distance():
                    bounding_box.draw(cv2=cv2, image=image)

    try:
        for i in range(len(bounding_boxes)):
            for j in range(i + 1, len(bounding_boxes)):
                closest_box1, closest_box2 = bounding_boxes[i], bounding_boxes[j]
                distance_between = closest_box1.calculate_real_distance_to(closest_box2)
                
                if closest_box1.distance < env.DISTANCE_BETWEEN_OBJECT_AND_CAMERA_THRESHOLD or \
                    closest_box2.distance < env.DISTANCE_BETWEEN_OBJECT_AND_CAMERA_THRESHOLD or \
                    distance_between > env.SAFE_SPACE_THRESHOLD:
                    continue
                
                closest_box1.draw(cv2, image)
                closest_box2.draw(cv2, image)

                cv2.line(image, 
                        (int(closest_box1.center[0]), int(closest_box1.center[1])), 
                        (int(closest_box2.center[0]), int(closest_box2.center[1])), 
                        env.UNSAFE_COLOR, 2)
                
                mid_point = ((closest_box1.center[0] + closest_box2.center[0]) // 2, 
                            (closest_box1.center[1] + closest_box2.center[1]) // 2)
                cv2.putText(image, f"Distance: {distance_between:.2f}m", 
                            (int(mid_point[0]), int(mid_point[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, env.DISTANCE_TEXT_COLOR, 2)
                
    except:
        pass
    finally:
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