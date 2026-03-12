# DeepGuard AI: Multimodal Deepfake Detection & Forensic Suite

## 🚀 Project Overview
DeepGuard AI is a comprehensive, enterprise-grade forensic software suite designed to identify digital media manipulations across both **video and audio** modalities. Developed as a high-impact B.Tech AI & Data Science project at CGC University, it bridges the gap between state-of-the-art deep learning detection and relational data management to provide a verifiable "chain of custody" for digital evidence.

Unlike standard deepfake detectors, DeepGuard AI utilizes a hybrid **Spatio-Temporal + Acoustic** approach, supplemented with strict Database Isolation, Admin Portals, Explainable AI (XAI), and Certified Forensic Report generation.

## ✨ Core Features

### 1. Hybrid Visual Detection Engine
- **Facial Extraction:** Uses MTCNN for precise, margin-optimized facial cropping.
- **Deep Learning Model:** Evaluates frames using a custom PyTorch architecture (EfficientNet-backbone).
- **Forensic Heuristics:** Analyzes image noise distribution, Fast Fourier Transform (FFT) frequencies, laplacian texture smoothness, and color uniformity to catch manipulations that bypass standard CNNs.
- **Explainable AI (XAI):** Generates **Grad-CAM heatmaps** to visually highlight which facial regions triggered the AI.

### 2. Spectral Audio Forensics (New!)
- Uses `librosa` and `ffmpeg` to extract and analyze the video's audio track.
- Detects synthesized AI voices by analyzing **Spectral Flatness, MFCC Variance, Pitch (F0) Consistency, and Zero-Crossing Rates**.

### 3. Enterprise User Management
- **Role-Based Access Control (RBAC):** Distinct `user` and `admin` roles.
- **Database Isolation:** Users only see their own scan history.
- **Admin Portal:** Administrators have a macro-view of all system users, total scans, and average confidence metrics across the entire platform.

### 4. Certified Forensic Reporting
- **PDF Export:** Generates downloadable, certified PDF reports detailing the Trace ID, SHA-256 Hash, AI Engine used, and Final Verdict.
- **CSV Data Export:** Allows bulk export of scan logs for external analysis.
- **SHA-256 Chain of Custody:** Automatically hashes all uploaded files to maintain rigorous evidence integrity.

### 5. Advanced UI Capabilities
- **Time-Series Analysis:** Visual timeline chart of confidence scores across a video's duration.
- **Comparison Mode:** Side-by-side video analysis allowing investigators to compare a suspected fake directly against an authentic baseline.
- **Image Scanning:** Dedicated image pipeline for static photo verification.

## 🛠️ Technology Stack
- **Language:** Python 3.12
- **Deep Learning / CV:** PyTorch, Torchvision, FaceNet-PyTorch (MTCNN), OpenCV, PyTorch-Grad-CAM
- **Audio Analysis:** Librosa, FFmpeg
- **Backend Frame:** Flask (Werkzeug Secure User Sessions)
- **Database:** MySQL 8.0
- **Frontend:** HTML5/CSS3 (Jinja2 Templates, Chart.js)

## 📂 Project Directory Structure
```text
DeepGuard_AI/
├── app/
│   ├── app.py              # Main Flask Application
│   ├── templates/          # Jinja2 HTML Templates (Admin, Dashboard, Compare)
│   └── static/             # CSS & JS assets
├── database/               
│   ├── schema.sql          # MySQL schemas (Users, Media, Analysis, Artifacts)
│   └── db_manager.py       # Python-MySQL CRUD abstraction
├── models/                 # Saved PyTorch weights (.pth)
├── src/                    
│   ├── audio_analysis.py   # Librosa spectral audio analysis
│   ├── preprocess.py       # Data Pipeline
│   ├── model_arch.py       # Core AI Architecture
│   └── train.py            # Training routines
├── requirements.txt        # Dependencies
└── README.md               
```

## 🔧 Installation & Setup

### 1. Requirements
Ensure you have **MySQL 8.0+** and **FFmpeg** installed on your system.
- *FFmpeg (Windows):* `winget install ffmpeg`

### 2. Clone the Repository
```bash
git clone https://github.com/sauravt20250405/DeepGuardAI.git
cd DeepGuardAI
```

### 3. Set up the Database
Import the SQL schema into your local MySQL instance:
```bash
mysql -u root -p < database/schema.sql
```
*(Ensure you update the database credentials in `app/app.py` and `database/db_manager.py` if your local root password differs from the script defaults).*

### 4. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 5. Run the Application
```bash
python app/app.py
```
The application will start on `http://localhost:5000`. 
- **Default Admin Account:** Username: `admin` | Password: `admin123`

## ✅ Project Status
- [x] Initial Architecture & Database Schema
- [x] Spatio-Temporal Model Integration
- [x] Web Dashboard with XAI (Grad-CAM)
- [x] User Authentication & Admin Portal
- [x] Audio Spectral Analysis
- [x] Comparison Mode
- [x] Forensic PDF & CSV Export Tools