import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import os
import pandas as pd
from datetime import datetime
import plotly.express as px
import io
import base64

# Configure page
st.set_page_config(
    page_title="PPE Detection System",
    page_icon="🦺",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .compliant {
        background-color: #28a745;
        padding: 10px;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .partial {
        background-color: #ffc107;
        padding: 10px;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .non-compliant {
        background-color: #dc3545;
        padding: 10px;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class PPEDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.classes = ['Hardhat', 'Mask', 'Safety Vest', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']
        
    
    def get_statistics(self, detections):
        stats = {
            'total_detections': len(detections),
            'proper_ppe': len([d for d in detections if 'NO-' not in d['class']]),
            'violations': len([d for d in detections if 'NO-' in d['class']]),
            'detection_classes': {}
        }
        
        for det in detections:
            cls = det['class']
            if cls not in stats['detection_classes']:
                stats['detection_classes'][cls] = 0
            stats['detection_classes'][cls] += 1
            
        return stats

    def get_download_link(self, img):
        """Generate a download link for an image"""
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:file/png;base64,{img_str}" download="ppe_detection.png">Download Result</a>'
        return href

    def detect_ppe(self, image, conf=0.5):
        results = self.model(image, conf=conf)[0]
        detections = []
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            
            detections.append({
                'class': self.classes[class_id],
                'confidence': confidence,
                'bbox': bbox
            })
            
        return detections

    def analyze_compliance(self, detections):
        ppe_status = {
            'Hardhat': False,
            'Mask': False,
            'Safety Vest': False
        }
        
        for det in detections:
            if det['class'] in ppe_status:
                ppe_status[det['class']] = True
                
        items_present = sum(ppe_status.values())
        
        if items_present == 3:
            return "🟩 Compliant", "compliant"
        elif items_present > 0:
            return "🟨 Partially Compliant", "partial"
        else:
            return "🟥 Non-Compliant", "non-compliant"

    def draw_detections(self, image, detections):
        img = np.array(image.copy())
        
        # Convert PIL to CV2 format if necessary
        if isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        for det in detections:
            bbox = det['bbox'].astype(int)
            label = f"{det['class']} {det['confidence']:.2f}"
            
            # Color based on PPE type (Green for proper PPE, Red for violations)
            if 'NO-' in det['class']:
                color = (0, 0, 255)  # Red for violations
            else:
                color = (0, 255, 0)  # Green for proper PPE
                
            # Draw bounding box
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            cv2.putText(img, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    st.title("🦺 PPE Detection System")
    st.write("Upload an image to detect PPE compliance")

    # Initialize model and detector
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = "yolov8n.pt"
        st.warning("⚠️ Custom model not found, using YOLOv8n base model")
    
    @st.cache_resource
    def load_model():
        return PPEDetector(MODEL_PATH)
    
    detector = load_model()
    
    # Add sidebar options with unique keys
    st.sidebar.markdown("## ⚙️ Settings")
    confidence = st.sidebar.slider(
        "Detection Confidence", 
        0.0, 
        1.0, 
        0.5,
        key="confidence_slider"
    )
    show_stats = st.sidebar.checkbox("Show Statistics", True, key="show_stats_checkbox")
    enable_history = st.sidebar.checkbox("Enable History", True, key="enable_history_checkbox")
    
    # Initialize session state for history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Add tabs
    tab1, tab2, tab3 = st.tabs(["📸 Detection", "📊 Statistics", "📜 History"])
    
    with tab1:
        # File uploader with unique key
        uploaded_file = st.file_uploader(
            "Choose an image", 
            type=['jpg', 'jpeg', 'png'],
            key="image_uploader"  # Add unique key
        )        
        if uploaded_file is not None:
            # Load the image
            image = Image.open(uploaded_file)
            
            if st.button("Detect PPE"):
                with st.spinner("Analyzing image..."):
                    detections = detector.detect_ppe(image, confidence)
                    annotated_image = detector.draw_detections(image, detections)
                    compliance_text, compliance_class = detector.analyze_compliance(detections)
                    stats = detector.get_statistics(detections)
                    
                    # Save to history
                    if enable_history:
                        st.session_state.history.append({
                            'timestamp': datetime.now(),
                            'compliance': compliance_text,
                            'detections': len(detections),
                            'violations': stats['violations']
                        })
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Image")
                        st.image(image, use_column_width=True)
                    
                    with col2:
                        st.subheader("Detection Results")
                        st.image(annotated_image, use_column_width=True)
                        st.markdown(detector.get_download_link(Image.fromarray(annotated_image)), unsafe_allow_html=True)
                    
                    # Display compliance status
                    st.markdown(f"<div class='{compliance_class}'><h3>{compliance_text}</h3></div>", 
                              unsafe_allow_html=True)
                    
                    if show_stats:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Detections", stats['total_detections'])
                        col2.metric("Proper PPE", stats['proper_ppe'])
                        col3.metric("Violations", stats['violations'])
                        
                        # Display class distribution
                        df = pd.DataFrame(list(stats['detection_classes'].items()), 
                                        columns=['Class', 'Count'])
                        fig = px.bar(df, x='Class', y='Count', 
                                   title='PPE Class Distribution',
                                   color='Class')
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if len(st.session_state.history) > 0:
            df = pd.DataFrame(st.session_state.history)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Scans", len(df))
            col2.metric("Avg Detections", f"{df['detections'].mean():.1f}")
            col3.metric("Total Violations", df['violations'].sum())
            
            # Timeline chart
            fig = px.line(df, x='timestamp', y=['detections', 'violations'],
                         title='Detection History Timeline')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if len(st.session_state.history) > 0:
            st.dataframe(pd.DataFrame(st.session_state.history))
        else:
            st.info("No detection history available yet.")

if __name__ == "__main__":
    main()