<h1 align="center">
  🚁 Real-Time Computer Vision & Deep Learning Drone Tracking System
</h1>

<p align="center">
  <b>AI-powered autonomous drone tracking with YOLOv8, gesture control, and manual override on DJI Tello</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Drone-DJI%20Tello-blue?style=for-the-badge&logo=dji" />
  <img src="https://img.shields.io/badge/Framework-YOLOv8-orange?style=for-the-badge&logo=opencv" />
  <img src="https://img.shields.io/badge/Deep%20Learning-PyTorch%20%7C%20TensorFlow-red?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Computer%20Vision-OpenCV-green?style=for-the-badge&logo=opencv" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" />
  <img src="https://img.shields.io/github/license/Narein-karthik/Real-Time-Computer-Vision-and-Deep-Learning-Based-Drone-Tracking-System-using-DJI-Tello?style=flat-square" />
  <img src="https://img.shields.io/github/languages/top/Narein-karthik/Real-Time-Computer-Vision-and-Deep-Learning-Based-Drone-Tracking-System-using-DJI-Tello?style=flat-square" />
</p>

---

## 🧠 Project Description

This project presents a **real-time AI-powered drone tracking system** built using **computer vision and deep learning** on the DJI Tello platform. The system integrates YOLOv8-based detection for autonomous tracking, intelligent control algorithms, and human–drone interaction for advanced navigation capabilities.

It supports **autonomous tracking**, **gesture-based control**, and **manual override**, allowing seamless switching between AI-driven navigation and user control. GPU acceleration (CUDA) is utilized for high-performance inference, enabling low-latency real-time tracking.

---

## ✨ Key Features

- 🎯 **YOLOv8-based real-time object/face tracking**  
- ✋ **Deep learning–based hand gesture control**  
- ⌨️ **Manual keyboard override for safety**  
- 🤖 **AI-driven autonomous navigation**  
- ⚡ **GPU acceleration (CUDA support)**  
- 📹 **Live video streaming and recording**  
- 🔁 **Multi-mode control (AI + Gesture + Manual)**  
- 🧩 **Modular, extensible architecture**

---

## 🏗️ System Architecture

The system is organized into layered modules:

### 🛰️ Perception Layer
- **YOLOv8** – Object/face detection  
- **MediaPipe** – Hand landmark detection  
- **OpenCV** – Vision processing

### 🧮 Intelligence Layer
- **PyTorch** – YOLO inference  
- **TensorFlow** – Gesture recognition model

### 🎮 Control & Interaction Layers
- **djitellopy** – DJI Tello SDK  
- Gesture recognition and manual keyboard control

### ⚙️ Acceleration Layer
- **CUDA** for GPU-accelerated inference

---

## 🎛️ Functional Modes

### 🤖 Autonomous Tracking Mode
- YOLOv8-based detection  
- Target locking and distance regulation  
- PID-style control logic

### ✋ Hand Gesture Control Mode
- Hand landmark extraction  
- Deep learning classification  
- Gesture-to-command mapping

### 🕹️ Manual Mode
- Keyboard override  
- Safety control and precise navigation

---

## 📁 Project Structure

```bash
ai-drone-tracking-system/
│
├── tracking/
│   └── face_tracking.py          # YOLOv8-based face/object tracking
│
├── gesture_control/
│   └── gesture_tracking.py       # Hand tracking and gesture control
│
├── models/
│   └── README.md                 # Model documentation
│
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 📦 Dependencies

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

## 🔧 Installation

```bash
git clone https://github.com/Narein-karthik/Real-Time-Computer-Vision-and-Deep-Learning-Based-Drone-Tracking-System-using-DJI-Tello.git
cd Real-Time-Computer-Vision-and-Deep-Learning-Based-Drone-Tracking-System-using-DJI-Tello
pip install -r requirements.txt
```

---

## 🚀 Execution

### 🎯 Face/Object Tracking Mode

```bash
python tracking/face_tracking.py
```

**Features:**
- Automatically locks onto a detected face/object  
- Adjusts drone orientation and distance using control logic

### ✋ Gesture Control Mode

```bash
python gesture_control/gesture_tracking.py
```

**Example gesture mapping:**
- Open palm: Takeoff  
- Closed fist: Land  
- Swipe left/right: Move left/right  
- Palm forward: Move forward/backward

---

## 🌍 Applications

- 🔐 **Autonomous surveillance systems**  
- 📡 **Smart monitoring solutions**  
- 🤝 **Human–robot interaction**  
- 🧪 **AI robotics research and education**  
- 🏙️ **Smart city robotics and automation**

---

## 🚧 Future Enhancements

- 👁️‍🗨️ Multi-object tracking with priority queue  
- 🐝 Swarm drone coordination  
- 🗣️ Voice-based control integration  
- 🧭 Autonomous path planning and obstacle avoidance  
- ☁️ Cloud + Edge AI optimization  
- 📱 Mobile app–based control interface

---

## 📚 Publications & Reports

### 📜 Published Research Paper

**Title:** AUTO TRACK: SMART AERIAL OBJECT TRACKING WITH DEEP LEARNING

**Authors:** Chethana G M, Dhanush M, Narein Karthik E, Nithin P  
**Advisor:** Dhivya R (Assistant Professor, AIML, Bangalore Technological Institute)

**Published in:** International Research Journal of Modernization in Engineering Technology and Science (IRJMETS)  
**Volume:** 07 | **Issue:** 09 | **Month:** September 2025  
**DOI:** [10.56726/IRJMETS83167](https://www.doi.org/10.56726/IRJMETS83167)  
**e-ISSN:** 2582-5208  
**Impact Factor:** 8.187  
**Publication Date:** October 1, 2025

**Journal Website:** [www.irjmets.com](http://www.irjmets.com)

### 🎯 Paper Highlights

- **Object Detection & Tracking:** YOLOv8n-based model optimized for drone vision with multi-object tracking
- **Obstacle Avoidance:** Computer vision with depth estimation and path correction algorithms
- **Gesture Recognition:** Reinforcement Learning CNN for real-time 3D hand gesture control
- **Flight Control:** PID-based control loop integration for stability and maneuverability
- **Safety Module:** 3D-printed detachable ducted propellers (10g weight) for enhanced flight stability

### 📄 Access Full Report

> 📎 **Download:** [Full Research Paper PDF](docs/IRJMETS70900072870-1st-publication-report.pdf)  
> 🎯 **Certificate:** [Publication Certificate PDF](docs/IRJMETS70900072870-4-1st-publication-certificate.pdf)

**Note:** Upload the PDF files to a `/docs` folder in your repository for the links to work.

---

## 👨‍💻 Authors

- **Narein Karthik E** – AI & ML Student (Computer Vision, Robotics)  
- **Nithin P** – AI & ML Student (Deep Learning, Control Systems)  
- **Dhanush M** – AI & ML Student (Integration & Testing)  
- **Chethana G M** – AI & ML Student (Human–Computer Interaction)

---

## 🙏 Acknowledgements

- DJI Tello SDK (djitellopy) for drone communication  
- Ultralytics YOLOv8 for real-time detection  
- Google MediaPipe for robust hand landmark detection

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">Made with ❤️ by the AI & ML Team at Bangalore Technological Institute</p>
