from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import torch
import tempfile
import os
from PIL import Image
from facenet_pytorch import MTCNN
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = FastAPI(title="DeepGuard Enterprise Inference (Multi-Frame)")

device = torch.device("cpu")
print(f"Booting Inference Engine on: {device}")

# --- 1. INITIALIZE FACE DETECTOR (MTCNN) ---
print("Loading MTCNN Face Extractor...")
mtcnn = MTCNN(keep_all=False, device=device)

# --- 2. LOAD THE EXPERT BRAIN (HUGGING FACE CLOUD) ---
print("Loading Expert AI from Hugging Face...")
MODEL_NAME = "dima806/deepfake_vs_real_image_detection"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

print(f"Verified Model Labels: {model.config.id2label}")
print("DeepGuard is Online and armed for Multi-Frame Analysis.")


def get_frames(video_path, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames > 0:
        step = max(1, total_frames // max_frames)
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            if len(frames) >= max_frames:
                break
    cap.release()
    return frames


@app.get("/")
def home():
    return {"status": "DeepGuard Enterprise is Online (Multi-Frame)"}


@app.post("/scan/")
async def scan_video(file: UploadFile = File(...)):
    print(f"\n--- New Multi-Frame Scan Request: {file.filename} ---")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(await file.read())
        temp_path = tfile.name

    try:
        # We will extract 5 frames.
        extracted_frames = get_frames(temp_path, max_frames=5)

        if not extracted_frames:
            os.unlink(temp_path)
            return {"error": "Could not read video file or extract frames."}

        frame_results = []
        frames_with_faces = 0

        for i, frame in enumerate(extracted_frames):
            print(f"  Analyzing Frame {i+1}...")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            boxes, _ = mtcnn.detect(pil_image)

            if boxes is None:
                continue

            frames_with_faces += 1
            box = [int(b) for b in boxes[0]]
            face_img = pil_image.crop(box)

            inputs = processor(images=face_img, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                # Get the probabilities of all classes
                probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

                # Find the index of the highest probability
                predicted_class_id = logits.argmax(-1).item()

                # Look up the actual text label for that index
                predicted_label = model.config.id2label[predicted_class_id]

                # Get the confidence percentage for that specific label
                confidence = probabilities[predicted_class_id].item() * 100

                frame_results.append({
                    "label": predicted_label,
                    "confidence": confidence
                })

        os.unlink(temp_path)

        if not frame_results:
            return {
                "filename": file.filename,
                "error": "No human faces detected in any of the extracted frames.",
                "verdict": "Unknown"
            }

        # --- CALCULATE FINAL VERDICT (Voting System) ---
        # Count how many frames were declared fake vs real
        fake_count = sum(1 for res in frame_results if "fake" in res['label'].lower())
        real_count = len(frame_results) - fake_count

        if fake_count >= real_count:
            final_verdict = "Fake"
            # Average the confidence of the frames that voted "Fake"
            if fake_count > 0:
                final_confidence = sum(res['confidence'] for res in frame_results if "fake" in res['label'].lower()) / fake_count
            else:
                 final_confidence = 0
        else:
            final_verdict = "Real"
             # Average the confidence of the frames that voted "Real"
            if real_count > 0:
                final_confidence = sum(res['confidence'] for res in frame_results if "real" in res['label'].lower()) / real_count
            else:
                final_confidence = 0

        print(f"Final Multi-Frame Result: {final_verdict} ({final_confidence:.2f}%) based on {frames_with_faces} frames.")

        # Clean up the output data for the JSON response
        breakdown = [f"{res['label']} ({res['confidence']:.1f}%)" for res in frame_results]

        return {
            "filename": file.filename,
            "faces_analyzed": frames_with_faces,
            "verdict": final_verdict,
            "confidence": round(final_confidence, 2),
            "frame_breakdown": breakdown
        }

    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return {"error": f"Internal Server Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)