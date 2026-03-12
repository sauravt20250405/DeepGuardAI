from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import torch
import os
import tempfile
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
import sys

# --- Path Configuration ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.append(PROJECT_ROOT)

from src.model_arch import DeepGuardModel

# Initialize the API Application
app = FastAPI(
    title="DeepGuard AI API",
    description="RESTful API for Spatio-Temporal Deepfake Detection",
    version="1.0.0"
)

# Load AI Components globally so they stay in memory
device = torch.device('cpu')
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, select_largest=True, post_process=False, device=device)
model = DeepGuardModel(pretrained=False)
model_path = os.path.join(PROJECT_ROOT, "models", "deepguard_best_model.pth")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_sharpness(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

@app.get("/")
def read_root():
    return {"status": "DeepGuard AI API is Online"}

@app.post("/scan/")
async def scan_video(file: UploadFile = File(...)):
    """
    Upload an MP4 video to run a full spatio-temporal deepfake scan.
    Returns a JSON payload with the verdict and confidence score.
    """
    if not file.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")
        
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            content = await file.read()
            tfile.write(content)
            temp_path = tfile.name
            
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames and score sharpness
        sample_count = 10
        frame_indices = [int(total_frames * (i / sample_count)) for i in range(1, sample_count)]
        extracted_faces = [] 
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = mtcnn(frame_rgb)
                
                if face is not None:
                    face_img = face.permute(1, 2, 0).numpy().astype('uint8')
                    sharpness = calculate_sharpness(face_img)
                    face_pil = Image.fromarray(face_img)
                    face_t = transform(face_pil)
                    extracted_faces.append((sharpness, face_t))
        
        cap.release()
        os.unlink(temp_path)
        
        if len(extracted_faces) == 0:
            return {"error": "No faces detected in the video."}
            
        # Sort by sharpness and grab the top 3
        extracted_faces.sort(key=lambda x: x[0], reverse=True)
        best_faces = [f[1] for f in extracted_faces[:3]]
        
        # Run the AI Inference
        batch = torch.stack(best_faces).to(device)
        with torch.no_grad():
            outputs = model(batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            avg_probs = torch.mean(probabilities, dim=0)
            fake_prob = avg_probs[0].item() * 100
            real_prob = avg_probs[1].item() * 100
            
        verdict = "Fake" if fake_prob > 50 else "Real"
        confidence = fake_prob if fake_prob > 50 else real_prob
        
        return {
            "filename": file.filename,
            "verdict": verdict,
            "confidence_score": round(confidence, 2),
            "model_engine": "EfficientNet-B0"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))