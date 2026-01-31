# Real-Time Computer Vision and Deep Learning Based Drone Tracking System using DJI Tello

## Project Description

This project presents a **real-time AI-powered drone tracking system** built using **computer vision and deep learning** on the DJI Tello platform. The system integrates YOLOv8-based deep learning models for real-time detection and tracking with intelligent control algorithms to enable autonomous navigation and human–drone interaction.

It supports **autonomous tracking**, **gesture-based control**, and **manual override**, allowing seamless switching between AI-driven navigation and user control. GPU acceleration (CUDA) is utilized for high-performance inference, enabling low-latency real-time tracking.

---

## Key Features

*  YOLOv8-based real-time object/face tracking
*  Deep learning–based gesture control system
*  Manual keyboard override
*  AI-driven autonomous navigation
*  GPU acceleration (CUDA support)
*  Live video streaming and recording
*  Multi-mode control (AI + Gesture + Manual)
*  Intelligent control logic
*  Modular system architecture

---

## System Architecture

**Perception Layer**

* YOLOv8 (object/face detection)
* MediaPipe (hand landmark detection)
* OpenCV (vision processing)

**Intelligence Layer**

* PyTorch (YOLO inference)
* TensorFlow (gesture recognition model)

**Control Layer**

* djitellopy (DJI Tello SDK)

**Interaction Layer**

* Gesture recognition
* Manual keyboard control

**Acceleration Layer**

* CUDA (GPU inference support)

---

## Project Structure

```
ai-drone-tracking-system/
│
├── tracking/
│   └── face_tracking.py
│
├── gesture_control/
│   └── gesture_tracking.py
│
├── models/
│   └── README.md
│
├── requirements.txt
└── README.md
```

---

## Dependencies

| Library       | Purpose                                                 |
| ------------- | ------------------------------------------------------- |
| djitellopy    | Communication and control interface for DJI Tello drone |
| opencv-python | Real-time video processing and visualization            |
| ultralytics   | YOLOv8 deep learning framework                          |
| torch         | GPU acceleration and deep learning inference            |
| tensorflow    | Gesture recognition model framework (.h5)               |
| mediapipe     | Hand landmark detection for gesture extraction          |
| numpy         | Numerical computation and preprocessing                 |
| keyboard      | Manual keyboard-based drone control                     |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Execution

### Face/Object Tracking Mode

```bash
python tracking/face_tracking.py
```

### Gesture Control Mode

```bash
python gesture_control/gesture_tracking.py
```

---

## Functional Modes

### Autonomous Tracking Mode

* YOLOv8-based detection
* Target locking
* PID-style control logic
* Distance regulation

### Hand Gesture Control Mode

* Hand landmark extraction
* Deep learning classification
* Gesture-command mapping
* Real-time drone response

### Manual Mode

* Keyboard override
* Safety control
* Manual navigation

---

## Applications

* Autonomous surveillance systems
* Smart monitoring solutions
* Human–robot interaction
* AI robotics research
* Intelligent navigation systems
* Smart city robotics
* AI-based automation platforms
* Research and academic project

---

## Academic Relevance

This project demonstrates practical integration of:

* Computer Vision
* Deep Learning
* Robotics
* Embedded AI
* Control Systems
* Human–Computer Interaction
* Real-time Systems
* Autonomous Systems

---

## Future Enhancements

* Multi-object tracking
* Swarm drone coordination
* Voice-based control
* Autonomous path planning
* Cloud AI integration
* Edge AI optimization
* Mobile app control


---

##  Author

**Narein Karthik E**
**Nithin P**
**Dhanush M**
**Cheathana G M**
AI & ML Students
Focus Areas: Artificial Intelligence, Robotics, Computer Vision, Autonomous Systems

---

## License

This project is developed for academic and research purposes.
