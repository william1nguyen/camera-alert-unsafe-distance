# AUTO-PARKING SYSTEM

## Abilities

- Object detection.
- Distance estimation.
- Auto Find Parking area and show guideline for parking.

## How to run this project ?

### Setup environment

```
$ conda create -f environment.yml
```

### Run project

- Run with camera

```
$ python main.py
```

- Add video `mp4` file into `tests` folder.
  E.g. tests/test1.mp4

```
$ python main.py --tests=tests/test1.mp4
```
