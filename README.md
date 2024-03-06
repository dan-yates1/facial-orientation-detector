# The Flash Pack - Interview Task

A real-time facial orientation detector using MediaPipe and OpenCV that assesses if a person is looking directly at the camera by analyzing 3D facial landmarks and Euler angles. It calibrates to individual posture and camera setup, outputting the face's orientation status live.

## Screenshots

Here are some screenshots demonstrating the app in action:

![Imgur](https://i.imgur.com/JurshUx.jpeg "Looking Straight")

![Imgur](https://i.imgur.com/8tyAL5g.jpeg "Not Looking Straight")

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

- Python 3.9
- OpenCV
- MediaPipe
- Transforms3d

#### Setup

1. **Create and Activate a Virtual Environment (Optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run Main**

   ```bash
   python main.py
   ```

## Built With

- [OpenCV](https://opencv.org/) - The computer vision library used
- [MediaPipe](https://google.github.io/mediapipe/) - Framework for building multimodal applied machine learning pipelines
- [Transforms3D](https://matthew-brett.github.io/transforms3d/) - Python library for 3D coordinate transformations

## Acknowledgements

- [Real-Time Head Pose Estimation FaceMesh with MediaPipe and OpenCV: A Comprehensive Guide](https://medium.com/@jaykumaran2217/real-time-head-pose-estimation-facemesh-with-mediapipe-and-opencv-a-comprehensive-guide-b63a2f40b7c6) - Used code for drawing coordinates
- [Python Face Detection and Face Mesh (python OpenCV & MediaPipe package)](https://www.youtube.com/watch?v=yvXPKfil1hY) - Support for setting up the project
