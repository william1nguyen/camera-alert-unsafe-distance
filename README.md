# AUTO-PARKING SYSTEM

## Abilities

- Object detection.
- Distance estimation.
- Auto Find Parking area and show guideline for parking.

## How to run this project ?

```
$ conda create -f environment.yml
```

- With `MiDas Model` for depth estimation and pinhole camera model for calculating distance using focal length, real width, estimated width (pixel):

```
$ python object_detect_z.py
```

- Without `MiDas Model`:

```
$ python object_detect.py
```
