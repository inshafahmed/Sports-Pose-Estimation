# AI-Driven Sports Pose Estimation & Player Analytics

### Project Overview
This project implements a high-performance computer vision pipeline using **YOLOv8-Pose** to detect players and estimate skeletal keypoints across various sports (Cricket, Football, Rugby).

### Key Features
* **Multi-Sport Detection:** Tested on 6 distinct sports scenarios.
* **Real-Time Performance:** Achieves ~30+ FPS on NVIDIA T4 hardware.
* **Automated Metrics:** Generates frame-by-frame latency and crowd density logs.

### Technical Implementation
- **Framework:** Ultralytics YOLOv8
- **Model:** `yolov8n-pose.pt` (Nano)
- **Tracking:** ByteTrack
- **Hardware:** Google Colab (Tesla T4 GPU)

### How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Place videos in the `data/input/` folder.
3. Execute the script: `python src/main_inference.py`

### Performance Summary
The model demonstrates high robustness in foreground player detection with a confidence range of **0.5 - 0.9**. Metrics are exported to `Performance_Report.xlsx` for further statistical analysis.
