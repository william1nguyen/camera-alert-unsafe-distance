# Camera Alert Unsafe Distance System

A computer vision system that detects objects, estimates their distance, and alerts when objects are at an unsafe proximity using YOLOv11 from Ultralytics.

## Features

- Real-time object detection using Ultralytics YOLO
- Accurate distance estimation
- Configurable safety threshold alerts
- Support for both live camera feed and video file input

## Demo

https://github.com/user-attachments/assets/85365a95-b41b-46b0-aadb-fc78aa8c20be

## Installation

### Setup Environment

Create a conda environment using the provided configuration:

```bash
conda create -f environment.yml
```

### Activate Environment

```bash
conda activate yolo
```

## Usage

### Run with Live Camera

To use your computer's camera for real-time detection:

```bash
python main.py
```

### Run with Test Video

Place test videos in the `tests` folder, then specify the file path:

```bash
python main.py --tests=tests/<TEST_VIDEO>
```

## Requirements

All dependencies are listed in the `environment.yml` file.

## License

[MIT License](LICENSE)
