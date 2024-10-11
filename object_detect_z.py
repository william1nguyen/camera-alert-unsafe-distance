# Yolov8 for detecting objects
# MiDas for estimating object depth
# Pin Hole camera model to estimate distance

import cv2
import torch
import numpy as np
from errors import OpenCameraException
from models.yolov8 import yolov8
from env import *

object_detect_model = yolov8

def load_midas_model():
    # model_type = "MiDaS_small"
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    return midas, transform, device

def estimate_depth(image, midas, transform, device):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    return depth_map

def predict(frame):
    results = object_detect_model.track(frame, stream=True)
    return results

def calculate_distance(focal_length, real_object_width, pixel_width):
    distance = (real_object_width * focal_length) / pixel_width
    return distance

def check_safe_distance(distance):
    return distance > SAFE_DISTANCE_THRESHOLD

def depth_to_distance(depth):
    # This is a simple heuristic. You may need to adjust these values based on your specific camera and scene.
    min_distance = 0.5  # Minimum expected distance in meters
    max_distance = 10.0  # Maximum expected distance in meters
    
    # Invert depth since smaller depth values correspond to larger distances
    inverted_depth = 1.0 - depth
    
    # Map the inverted depth to the distance range
    estimated_distance = min_distance + (inverted_depth * (max_distance - min_distance))
    
    return estimated_distance

# Estimated distance from z_midas and focal_distance
def combine_distances(z_midas, focal_distance, weight=0.7):
    return (weight * z_midas) + ((1 - weight) * focal_distance)

def show_webcam_results(image, predicted_results, target_classes, midas_model, midas_transform, midas_device):
    depth_map = estimate_depth(image, midas_model, midas_transform, midas_device)

    for r in predicted_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            confidence = box.conf[0]

            if cls in target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                distance_focal = calculate_distance(
                    DEFAULT_FOCAL_LENGTH,
                    object_detect_model.object_height.get(cls),
                    abs(x2 - x1)
                )

                object_depth = np.mean(depth_map[y1:y2, x1:x2])
                # estimated_distance = depth_to_distance(object_depth)
                estimated_distance = combine_distances(object_depth, distance_focal)

                if estimated_distance > SAFE_DISTANCE_THRESHOLD:
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
                
                cv2.putText(
                    img=image,
                    text=f"{object_detect_model.target_classes[cls]} - D:{estimated_distance:.2f}m",
                    org=(x1, y1 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.9,
                    color=bounding_color,
                    thickness=2
                )

    # Create a more visually appealing depth map
    colored_depth_map = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=255), cv2.COLORMAP_INFERNO)
    
    # Resize depth map to match the original image size
    colored_depth_map_resized = cv2.resize(colored_depth_map, (image.shape[1], image.shape[0]))
    
    # Combine original image and depth map
    combined_image = cv2.addWeighted(image, 0.7, colored_depth_map_resized, 0.3, 0)

    cv2.imshow('Webcam with Distance Estimation', combined_image)

def start_webcam(cap, midas_model, midas_transform, midas_device):
    if not cap.isOpened():
        raise OpenCameraException()

    while True:
        ret, frame = cap.read()
        if ret:
            predicted_results = predict(frame=frame)
            target_classes = list(object_detect_model.target_classes.keys())
            show_webcam_results(
                image=frame, 
                predicted_results=predicted_results,
                target_classes=target_classes,
                midas_model=midas_model, 
                midas_transform=midas_transform, 
                midas_device=midas_device
            )
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def release(cap):
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    midas_model, midas_transform, midas_device = load_midas_model()
    # cap = cv2.VideoCapture("./tests/test1.mp4")
    cap = cv2.VideoCapture(0)
    try:
        start_webcam(cap=cap, midas_model=midas_model, midas_transform=midas_transform, midas_device=midas_device)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        release(cap=cap)