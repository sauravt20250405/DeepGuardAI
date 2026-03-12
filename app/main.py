import streamlit as st
import cv2
import torch
import os
import tempfile
import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
import sys
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from fpdf import FPDF

# --- Path Configuration ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.append(PROJECT_ROOT)

from src.model_arch import DeepGuardModel
from database.db_manager import DeepGuardDB

# --- Initialization ---
st.set_page_config(page_title="DeepGuard AI", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_system():
    device = torch.device('cpu')
    mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, select_largest=True, post_process=False, device=device)
    model = DeepGuardModel(pretrained=False)
    model_path = os.path.join(PROJECT_ROOT, "models", "deepguard_best_model.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        st.error("⚠️ Model weights not found! Please ensure you've run train.py.")
        
    return mtcnn, model, device

mtcnn, model, device = load_system()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_sharpness(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# --- UI Header ---
st.title("🛡️ DeepGuard AI")
st.markdown("### Spatio-Temporal Deepfake Detection & Forensic Logging")
st.markdown("---")

# --- Create Tabs ---
tab1, tab2 = st.tabs(["🔍 Live Forensic Scanner", "🗄️ MySQL Database Logs"])

# ==========================================
# TAB 1: THE SCANNER
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("1. Upload Media")
        uploaded_video = st.file_uploader("Upload a suspect .mp4 video", type=["mp4"])
        
        if uploaded_video is not None:
            st.video(uploaded_video)

    with col2:
        st.header("2. Forensic Analysis")
        
        if uploaded_video is not None:
            if st.button("Run DeepGuard Scan", use_container_width=True):
                with st.spinner("Isolating and scoring facial spatial data for sharpness..."):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_video.read())
                    tfile.close()
                    
                    cap = cv2.VideoCapture(tfile.name)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
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
                                extracted_faces.append((sharpness, face_t, face_img, idx))
                    
                    cap.release()
                    os.unlink(tfile.name)
                    
                    if len(extracted_faces) > 0:
                        extracted_faces.sort(key=lambda x: x[0], reverse=True)
                        best_faces = extracted_faces[:3]
                        
                        faces_tensor = [f[1] for f in best_faces]
                        display_faces = [f[2] for f in best_faces]
                        best_indices = [f[3] for f in best_faces]
                        
                        st.success(f"Successfully extracted the {len(best_faces)} sharpest facial anchors.")
                        st.image(display_faces, width=150, caption=[f"Frame {idx}" for idx in best_indices])
                        
                        with st.spinner("Running EfficientNet-B0 Spatio-Temporal Analysis & Generating Heatmaps..."):
                            batch = torch.stack(faces_tensor).to(device)
                            
                            # 1. Standard Prediction
                            with torch.no_grad():
                                outputs = model(batch)
                                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                                
                                avg_probs = torch.mean(probabilities, dim=0)
                                fake_prob = avg_probs[0].item() * 100
                                real_prob = avg_probs[1].item() * 100
                            
                            # 2. Grad-CAM Heatmap Generation (Explainable AI)
                            # We target the final convolutional layer of EfficientNet to see what it "sees"
                            target_layers = [model.backbone.features[-1]]
                            cam = GradCAM(model=model, target_layers=target_layers)
                            
                            # Generate a heatmap for the sharpest face (the first one in our sorted list)
                            grayscale_cam = cam(input_tensor=batch[0:1])[0, :]
                            
                            # Prepare the original image to overlay the heatmap
                            # Normalize pixel values to [0, 1] for the Grad-CAM utility
                            rgb_img = display_faces[0].astype(np.float32) / 255.0
                            heatmap_visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                                
                            st.markdown("---")
                            st.subheader("Verdict & Explainable AI Analysis")
                            
                            # Display the AI's "Thought Process"
                            st.image(heatmap_visualization, width=300, caption="Grad-CAM Forensic Saliency Map (Red = High Manipulation Probability)")
                            
                            if fake_prob > 50:
                                st.error(f"🚨 DEEPFAKE DETECTED (Confidence: {fake_prob:.2f}%)")
                                st.markdown("*Notice the highlighted thermal regions above showing the spatial artifacts that triggered the detection.*")
                            else:
                                st.success(f"✅ AUTHENTIC MEDIA (Confidence: {real_prob:.2f}%)")
                                st.markdown("*The thermal map shows uniform spatial consistency, indicating no digital manipulation.*")
                                
                            # --- DATABASE INGESTION LOGIC ---
                            try:
                                # IMPORTANT: Ensure your password is correct here!
                                db = DeepGuardDB(user='root', password='Luck@492025', database='DeepGuard_AI')
                                file_size_mb = uploaded_video.size / (1024 * 1024)
                                verdict_text = 'Fake' if fake_prob > 50 else 'Real'
                                confidence = fake_prob if fake_prob > 50 else real_prob
                                
                                media_id = db.log_media(uploaded_video.name, round(file_size_mb, 2))
                                analysis_id = db.save_analysis(media_id, round(confidence, 2), verdict_text, "EfficientNet-B0 + XAI")
                                
                                for frame_num in best_indices:
                                    db.log_artifact(analysis_id, frame_num, "Sharp Facial Spatial Scan")
                                    
                                db.close()
                                st.success(f"💾 Forensic data permanently logged to MySQL! (Media ID: {media_id})")
                                
                                # --- PDF REPORT GENERATOR ---
                                # 1. Build the PDF document
                                pdf = FPDF()
                                pdf.add_page()
                                pdf.set_font("helvetica", "B", 16)
                                pdf.cell(0, 10, "DeepGuard AI - Forensic Security Report", ln=True, align="C")
                                pdf.ln(10)
                                
                                pdf.set_font("helvetica", "", 12)
                                pdf.cell(0, 10, f"File Analyzed: {uploaded_video.name}", ln=True)
                                pdf.cell(0, 10, f"Database Media ID: {media_id}", ln=True)
                                pdf.cell(0, 10, f"AI Verdict: {verdict_text}", ln=True)
                                pdf.cell(0, 10, f"Confidence Score: {confidence:.2f}%", ln=True)
                                pdf.cell(0, 10, f"Detection Engine: EfficientNet-B0 + Grad-CAM Spatial Analysis", ln=True)
                                
                                pdf.ln(10)
                                pdf.multi_cell(0, 10, "Disclaimer: This report is automatically generated by DeepGuard AI. Results are based on spatio-temporal artifact detection and should be verified by human forensic experts in legal contexts.")
                                
                                # 2. Convert to byte stream for Streamlit download
                                pdf_bytes = bytes(pdf.output())
                                
                                # 3. Display the Download Button
                                st.download_button(
                                    label="📄 Download Official PDF Report",
                                    data=pdf_bytes,
                                    file_name=f"DeepGuard_Forensic_Report_ID{media_id}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                                
                            except Exception as e:
                                st.error(f"⚠️ Database Error: {e}")
                    else:
                        st.warning("No faces detected in the video sample.")

# ==========================================
# TAB 2: THE FORENSIC DATABASE DASHBOARD
# ==========================================
with tab2:
    st.header("🗄️ System-Wide Forensic Dashboard")
    st.markdown("Live analytics and forensic traces from the DeepGuard MySQL database.")
    
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
        
    try:
        # IMPORTANT: Keep your MySQL password here
        db = DeepGuardDB(user='root', password='Luck@492025', database='DeepGuard_AI')
        logs = db.get_all_logs()
        db.close()
        
        if logs:
            df = pd.DataFrame(logs)
            
            # --- 1. TOP LEVEL METRICS (KPIs) ---
            st.markdown("### 📊 Live System Analytics")
            
            total_scans = len(df)
            total_fakes = len(df[df['verdict'] == 'Fake'])
            total_reals = len(df[df['verdict'] == 'Real'])
            avg_conf = df['confidence_score'].mean()
            
            # Create 4 stylized metric cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Media Scanned", total_scans)
            m2.metric("Deepfakes Detected 🚨", total_fakes)
            m3.metric("Authentic Media ✅", total_reals)
            m4.metric("Avg. AI Confidence", f"{avg_conf:.1f}%")
            
            st.markdown("---")
            
            # --- 2. DATA VISUALIZATION CHARTS ---
            st.markdown("### 📈 Detection Distribution")
            chart_col1, chart_col2 = st.columns([1, 2])
            
            with chart_col1:
                st.markdown("**Fake vs. Real Ratio**")
                # Create a simple bar chart comparing verdicts
                verdict_counts = df['verdict'].value_counts()
                st.bar_chart(verdict_counts, color="#FF4B4B") 
                
            with chart_col2:
                st.markdown("**Confidence Scores Over Time**")
                # Create a scatter chart mapping every scan's confidence score
                st.scatter_chart(df, x='upload_date', y='confidence_score', color='verdict')
                
            st.markdown("---")
            
            # --- 3. THE RAW DATA TABLE ---
            st.markdown("### 📋 Raw Forensic Trace Logs")
            
            # Clean up the column names for a professional look
            df_display = df.rename(columns={
                'media_id': 'Media ID',
                'filename': 'File Name',
                'file_size_mb': 'Size (MB)',
                'upload_date': 'Timestamp',
                'confidence_score': 'Confidence %',
                'verdict': 'Verdict',
                'model_used': 'AI Engine'
            })
            
            # Display the interactive table
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
        else:
            st.info("The database is currently empty. Run a scan to generate forensic logs!")
            
    except Exception as e:
        st.error(f"Could not connect to MySQL. Ensure the database is running. (Error: {e})")