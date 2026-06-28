# Face Recognition using DLib

## About

This is a simple face recognition project developed using Python, DLib, OpenCV and the face_recognition library. The project detects a person's face through the webcam and identifies them if their images are available in the dataset.

## Features

* Detect faces using webcam
* Recognize known people
* Show "Unknown" for new faces
* Real-time face recognition

## Technologies Used

* Python
* OpenCV
* DLib
* face_recognition
* NumPy

## Dataset

The dataset contains images of:

* Arnold Schwarzenegger
* Colin Powell
* Michael Jackson

More people can be added by creating a new folder inside the dataset and placing their images in it.

## How to Run

1. Install the required packages.

```
pip install -r requirements.txt
```

2. Run the program.

```
python main.py
```

3. The webcam will open and recognize the faces in the dataset.

## Project Output

* Detects faces in real time
* Recognizes known faces
* Displays "Unknown" for faces not present in the dataset

## Future Work

* Add more people to the dataset
* Improve recognition accuracy
* Support video file recognition

## Author

Dinesh Kumar

GitHub: https://github.com/dineshbadger
