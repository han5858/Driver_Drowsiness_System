# 😴 AI Driver Drowsiness Detection System

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face_Mesh-orange?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=for-the-badge)

## 📖 Overview
This project is a real-time safety system designed to prevent accidents caused by driver fatigue. It uses **Computer Vision** and **Facial Landmarking** to monitor the driver's eyes.

Using **MediaPipe Face Mesh**, the system tracks the eye aspect ratio (EAR). If the eyes remain closed for a specific duration (simulating drowsiness or sleep), an **Audio Alarm** is triggered to wake the driver.

---

## ⚙️ How It Works (The Math)
The system calculates the **Eye Aspect Ratio (EAR)** based on 6 facial landmarks per eye.

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

* **Vertical Distance:** Distance between the upper and lower eyelids.
* **Horizontal Distance:** Distance between the left and right corners of the eye.
* **Threshold:** If EAR falls below `0.26`, the eye is considered **CLOSED**.

---

## ✨ Key Features
* **Real-Time Tracking:** Processes video frames instantly using CPU (no heavy GPU required).
* **Face Mesh Technology:** Uses Google MediaPipe to map 468 facial landmarks.
* **Audio Alert:** Triggers a Windows system beep (Alarm) when drowsiness is detected.
* **Visual Warning:** Displays "DROWSINESS DETECTED" warning on the screen.

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/han5858/Driver-Drowsiness-Detection.git](https://github.com/han5858/Driver-Drowsiness-Detection.git)
cd Driver-Drowsiness-Detection
