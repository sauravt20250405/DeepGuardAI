# DeepGuard AI: Spatio-Temporal Deepfake Detection & Forensic Logging

## 🚀 Project Overview
DeepGuard AI is a high-impact forensic software suite designed to identify digital video manipulations. Developed as a B.Tech AI & Data Science academic project at CGC University, Mohali, it bridges the gap between deep learning-based detection and relational data management for legal/forensic tracking.

The system utilizes a Spatio-Temporal approach to analyze facial inconsistencies and logs every scan with relational integrity to provide a verifiable "chain of custody" for digital evidence.

## 🛠️ Technology Stack
- **Language:** Python 3.12
- **Deep Learning:** PyTorch, Torchvision
- **Computer Vision:** OpenCV, FaceNet-PyTorch (MTCNN), MediaPipe
- **Database:** MySQL 8.0 (Relational Forensic Traces)
- **Frontend:** Streamlit
- **Infrastructure:** Optimized for Intel i5-12450H | 16GB RAM

## 📂 Project Directory Structure
```text
DeepGuard_AI/
├── app/                    # Web interface (Streamlit)
│   └── main.py             # Main application entry point
├── data/
│   ├── raw/                # Original DFDC .mp4 files
│   └── processed/          # 224x224 Face Crops (Train/Val)
├── database/               # DBMS Module
│   ├── schema.sql          # MySQL table definitions
│   └── db_manager.py       # Python-MySQL CRUD logic
├── models/                 # Saved PyTorch weights (.pth)
├── notebooks/              # Research & Experimentation
├── src/                    # Core AI Logic
│   ├── preprocess.py       # Lossy frame extraction script
│   ├── dataset.py          # Custom PyTorch DataLoader
│   ├── model_arch.py       # CNN + LSTM Architecture
│   └── train.py            # Training & Validation loops
├── requirements.txt        # Dependency list
└── README.md               # Project documentation
```

Target Machine: Honor MagicBook X16 (2024)

CPU Optimization: Leveraging the 8-core Intel i5-12450H for parallel preprocessing.

Storage Management (512GB SSD): To prevent disk overflow during training on the DFDC dataset, we implement a Lossy Extraction Strategy:

Temporal Sampling: 1 Frame Per Second (1 FPS) extraction.

Spatial Cropping: 224x224 pixel face-centered crops.

Format: High-efficiency JPEG storage for processed frames to save space.

📊 Database Schema Architecture
The system utilizes a MySQL backend to ensure all detections are queryable and forensic-ready:

MediaMetadata: Stores file fingerprints (filename, size, upload date).

AnalysisResults: Records AI verdicts (Real/Fake) and confidence scores.

DetectionArtifacts: Logs specific frame-level evidence (e.g., "Mismatched eye glint").

## 💾 Hardware Optimization Strategy
Target Machine: Honor MagicBook X16 (2024)

**CPU Optimization:** Leveraging the 8-core Intel i5-12450H for parallel preprocessing.

**Storage Management (512GB SSD):** To prevent disk overflow during training on the DFDC dataset, we implement a Lossy Extraction Strategy:
- Temporal Sampling: 1 Frame Per Second (1 FPS) extraction.
- Spatial Cropping: 224x224 pixel face-centered crops.
- Format: High-efficiency JPEG storage for processed frames to save space.

## 🔧 Installation & Setup

### Clone the Repository:
[x] Project Architecture & Directory Setup

[x] Database Schema Design

[x] Data Preprocessing (MTCNN Implementation)

[x] Model Training (ResNet/EfficientNet backbone)

[x] Streamlit Dashboard Integration