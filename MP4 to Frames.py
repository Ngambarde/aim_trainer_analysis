import cv2
import os
import glob


# --- CONFIGURATION ---
VIDEO_FOLDER = "videos/"
OUTPUT_FOLDER = "images/"
frame_interval = 30 # ie. 30 in a 30 fps video = capture an image every 1 second, 15 = capture image every 0.5 seconds, etc.
video_count = 0
total_videos = len(glob.glob1(VIDEO_FOLDER, "*.mp4"))

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

video_files = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))

for video_path in video_files:
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(OUTPUT_FOLDER, f"frame_{video_count}_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    video_count += 1
    print(f"Video {video_count} of {total_videos} has been extracted.")
    print(f"Saved {saved_count} frames from {video_name}")

print("Frame extraction Completed")

