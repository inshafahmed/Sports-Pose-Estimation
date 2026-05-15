import pandas as pd
import os
import cv2
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION & DIRECTORIES
# ==========================================
MODEL_NAME = 'yolov8n-pose.pt'
model = YOLO(MODEL_NAME)

INPUT_DIR = '/content/drive/MyDrive/Computer Vision Assignment/Resources/'
OUTPUT_DIR = './outputs/'
SCREENSHOT_DIR = './outputs/screenshots/'

# Create necessary folders
for folder in [OUTPUT_DIR, SCREENSHOT_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

video_files = [
    'video_1.mp4', 'video_2.mp4', 'video_3.mp4', 
    'video_4.mp4', 'video_5.mp4', 'video_6.mp4'
]

all_metrics = []

# ==========================================
# 2. MAIN PROCESSING LOOP
# ==========================================
print(f"Starting analysis on {len(video_files)} videos...")

for video in video_files:
    video_path = os.path.join(INPUT_DIR, video)
    if not os.path.exists(video_path):
        continue

    print(f"--- Processing: {video} ---")
    
    # Run tracking
    results = model.track(
        source=video_path, 
        persist=True, 
        save=True, 
        project=OUTPUT_DIR, 
        name=video.split('.')[0],
        conf=0.5, 
        stream=True
    )
    
    # Screenshot Management Variables
    video_name_clean = video.split('.')[0]
    captured_count = 0
    total_frames = 0
    
    # We store frames with detections to pick the best screenshots later
    detection_frames = []

    for frame_idx, r in enumerate(results):
        total_frames += 1
        num_players = len(r.boxes)
        
        # Log metrics for Excel
        latency = sum(r.speed.values())
        all_metrics.append({
            "Video_Name": video,
            "Frame_ID": frame_idx + 1,
            "Players_Detected": num_players,
            "Total_Latency_(ms)": latency,
            "Estimated_FPS": 1000 / latency if latency > 0 else 0
        })

        # Logic: If a person is detected, save the plotted frame as a candidate for screenshots
        if num_players > 0:
            # We take the annotated image from the results object
            annotated_frame = r.plot() 
            detection_frames.append(annotated_frame)

    # ==========================================
    # 3. SCREENSHOT EXTRACTION (2 per video)
    # ==========================================
    if len(detection_frames) >= 2:
        # Take a screenshot from the first 25% and one from the 75% point of detected sequences
        idx1 = len(detection_frames) // 4
        idx2 = (3 * len(detection_frames)) // 4
        
        shot1_path = f"{SCREENSHOT_DIR}{video_name_clean}_shot1.jpg"
        shot2_path = f"{SCREENSHOT_DIR}{video_name_clean}_shot2.jpg"
        
        cv2.imwrite(shot1_path, detection_frames[idx1])
        cv2.imwrite(shot2_path, detection_frames[idx2])
        print(f"Saved 2 screenshots for {video}")
    elif len(detection_frames) == 1:
        cv2.imwrite(f"{SCREENSHOT_DIR}{video_name_clean}_shot1.jpg", detection_frames[0])
        print(f"Saved 1 screenshot for {video}")

# ==========================================
# 4. EXPORT METRICS TO EXCEL
# ==========================================
df_metrics = pd.DataFrame(all_metrics)
with pd.ExcelWriter(f"{OUTPUT_DIR}Performance_Report.xlsx") as writer:
    df_metrics.to_excel(writer, sheet_name='Full_Frame_Log', index=False)

print(f"\nSUCCESS!")
print(f"1. Processed videos saved in: {OUTPUT_DIR}")
print(f"2. 12 Screenshots saved in: {SCREENSHOT_DIR}")
print(f"3. Excel metrics saved as: Performance_Report.xlsx")
