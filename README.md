# Real-Time Crowd Detection System

A real-time crowd detection and alert system built using **Python, YOLOv8, OpenCV, and PyTorch**.

The system captures live video from a webcam, detects objects in real time, counts people, and can trigger alerts when the crowd exceeds a predefined threshold.

## Features

* Real-time object detection using YOLOv8
* Person detection and counting
* Crowd threshold alert system
* FPS and object count display
* Screenshot capture using keyboard shortcuts
* CPU-based inference (no GPU required)

## Technologies Used

* Python 3.10+
* YOLOv8 (Ultralytics)
* OpenCV
* PyTorch
* NumPy
* Pillow

## Project Structure

```text
crowd_detection/
│
├── src/
│   ├── phase1_object_detection.py
│   ├── phase2_person_detection.py
│   └── phase3_crowd_alert.py
│
├── models/
├── outputs/
├── datasets/
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/real-time-crowd-detection-system.git
cd real-time-crowd-detection-system
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Usage

Run Phase 1:

```bash
python src/phase1_object_detection.py
```

Run Phase 2:

```bash
python src/phase2_person_detection.py
```

Run Phase 3:

```bash
python src/phase3_crowd_alert.py
```

## Controls

* **S** – Save screenshot
* **Q** – Quit application

## Results

* Detection Model: YOLOv8n
* Detectable Classes: 80
* Average FPS: 12–13 FPS (CPU)
* Resolution: 640 × 480
* Model Size: 6.2 MB
