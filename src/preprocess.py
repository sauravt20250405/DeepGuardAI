import cv2
import os
import json
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

# --- 1. Thermal Optimizations ---
# Limit PyTorch to 4 threads so it doesn't max out the i5-12450H
torch.set_num_threads(4) 

# --- 2. Robust Configuration & Paths ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "train_sample_videos")
META_PATH = os.path.join(RAW_DIR, "metadata.json")

# Output directories
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "train")
REAL_DIR = os.path.join(OUT_DIR, "real")
FAKE_DIR = os.path.join(OUT_DIR, "fake")
LOG_FILE = os.path.join(OUT_DIR, "processed_videos.log") # Checkpoint log

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

# Load checkpoint to resume progress
processed_videos = set()
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'r') as f:
        processed_videos = set(f.read().splitlines())

# --- 3. Initialize MTCNN ---
device = torch.device('cpu')
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, select_largest=True, post_process=False, device=device)

def process_video(video_filename, label):
    video_path = os.path.join(RAW_DIR, video_filename)
    if not os.path.exists(video_path):
        return

    save_dir = REAL_DIR if label == 'REAL' else FAKE_DIR
    video_prefix = video_filename.split('.')[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    frame_interval = int(fps) 
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # PRE-SCALING: Resize to max width of 640 to save CPU heat
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = mtcnn(frame_rgb)
            
            if face is not None:
                face_img = face.permute(1, 2, 0).numpy().astype('uint8')
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                out_path = os.path.join(save_dir, f"{video_prefix}_f{frame_count}.jpg")
                cv2.imwrite(out_path, face_img)
                
        frame_count += 1

    cap.release()
    
    # Write to log so we can resume if stopped
    with open(LOG_FILE, 'a') as f:
        f.write(video_filename + '\n')

if __name__ == "__main__":
    print("Loading metadata...")
    try:
        with open(META_PATH, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("Error: metadata.json not found!")
        exit()

    # Filter out already processed videos
    remaining_videos = {k: v for k, v in metadata.items() if k not in processed_videos}
    
    print(f"Total videos: {len(metadata)} | Already processed: {len(processed_videos)} | Remaining: {len(remaining_videos)}")
    print("Starting Thermal-Optimized Extraction...")
    
    for video_filename, data in tqdm(remaining_videos.items(), desc="Processing Videos", unit="vid"):
        label = data.get('label')
        process_video(video_filename, label)
        
    print("\nExtraction Complete!")