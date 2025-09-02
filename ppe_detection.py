import os
import sys

# 🔧 Fix Streamlit + Torch watcher bug - MUST be before streamlit import
os.environ["STREAMLIT_WATCHER_IGNORE"] = "tornado,torch"

import streamlit as st

# Streamlit page config - MUST be the first Streamlit command
st.set_page_config(page_title="PPE Detection", layout="wide", page_icon="🦺")

from PIL import Image
import pandas as pd
from datetime import datetime
import json
import tempfile

# Fix potential naming conflicts
if 'code' in sys.modules and hasattr(sys.modules['code'], '__file__'):
    if sys.modules['code'].__file__ and 'code.py' in sys.modules['code'].__file__:
        del sys.modules['code']

# Import YOLO after fixing conflicts
try:
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Error importing YOLO: {e}")
    st.error("Please rename your file from 'code.py' to something else (e.g., 'ppe_detection.py')")
    st.stop()

# Initialize session state for history
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# Load your trained model (update path if needed)
MODEL_PATH = "runs/detect/train/weights/best.pt"

# Load your trained model (update path if needed)
MODEL_PATH = "runs/detect/train/weights/best.pt"

@st.cache_resource
def load_model():
    try:
        # First try loading custom model
        if os.path.exists(MODEL_PATH):
            # Fix for PyTorch 2.6 weights_only issue
            import torch
            
            # Method 1: Add safe globals (Recommended)
            torch.serialization.add_safe_globals([
                'ultralytics.nn.tasks.DetectionModel',
                'ultralytics.nn.modules.block.C2f',
                'ultralytics.nn.modules.block.SPPF',
                'ultralytics.nn.modules.conv.Conv',
                'ultralytics.nn.modules.head.Detect'
            ])
            
            return YOLO(MODEL_PATH)
        else:
            # Fallback to pre-trained model for demo
            st.warning("⚠️ Custom model not found. Using YOLOv8n for demo.")
            return YOLO('yolov8n.pt')
        
    except Exception as e:
        st.error(f"Error with custom model: {e}")
        try:
            # Fallback to pre-trained model
            st.warning("⚠️ Loading YOLOv8n pre-trained model for demo.")
            return YOLO('yolov8n.pt')
            
        except Exception as e2:
            st.error(f"Failed to load any model: {e2}")
            raise e2

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.error("**Possible solutions:**")
    st.error("1. Update ultralytics: `pip install --upgrade ultralytics`")
    st.error("2. Make sure the model path is correct")
    st.error("3. Check if you have the latest PyTorch version")
    st.stop()

# PPE Requirements Configuration
PPE_REQUIREMENTS = {
    "safety_vest": {"required": True, "name": "Safety Vest"},
    "person": {"required": True, "name": "Person"},
    "gloves": {"required": True, "name": "Gloves"},
    "boots": {"required": True, "name": "Safety Boots"},
    "helmet": {"required": True, "name": "Hard Hat"}  # Now required
}

def analyze_compliance(detections):
    """Analyze PPE compliance based on detections"""
    detected_classes = [detection['class'].lower().replace(' ', '_').replace('-', '_') for detection in detections]
    
    # Create mapping for common variations of class names
    class_mapping = {
        'safety_vest': ['safety_vest', 'vest', 'high_vis', 'high_visibility_vest', 'reflective_vest'],
        'person': ['person', 'people', 'worker', 'human'],
        'gloves': ['gloves', 'glove', 'hand_protection'],
        'boots': ['boots', 'boot', 'safety_boots', 'work_boots', 'steel_toe'],
        'helmet': ['helmet', 'hard_hat', 'hardhat', 'safety_helmet', 'head_protection']
    }
    
    # Normalize detected classes
    normalized_detections = []
    for detected_class in detected_classes:
        for standard_name, variations in class_mapping.items():
            if detected_class in variations:
                normalized_detections.append(standard_name)
                break
        else:
            normalized_detections.append(detected_class)
    
    compliance_data = []
    required_items = 0
    detected_required = 0
    
    for class_name, config in PPE_REQUIREMENTS.items():
        is_detected = class_name in normalized_detections
        count = normalized_detections.count(class_name) if is_detected else 0
        
        compliance_data.append({
            "PPE Item": config["name"],
            "Required": "Yes" if config["required"] else "No",
            "Detected": "✅ Yes" if is_detected else "❌ No",
            "Count": count,
            "Status": "✅ Found" if is_detected else ("🔴 Missing" if config["required"] else "⚪ Not Required")
        })
        
        if config["required"]:
            required_items += 1
            if is_detected:
                detected_required += 1
    
    # Determine overall compliance
    compliance_percentage = (detected_required / required_items * 100) if required_items > 0 else 100
    
    if compliance_percentage == 100:
        compliance_status = "COMPLIANT"
        compliance_color = "🟢"
    elif compliance_percentage >= 50:
        compliance_status = "PARTIALLY COMPLIANT"
        compliance_color = "🟡"
    else:
        compliance_status = "NON-COMPLIANT"
        compliance_color = "🔴"
    
    return compliance_data, compliance_status, compliance_color, compliance_percentage

def process_detections(results):
    """Process YOLO results and extract detection information"""
    detections = []
    
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id] if class_id < len(model.names) else f"class_{class_id}"
                
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": box.xyxy[0].tolist()
                })
    
    return detections

def save_to_history(file_name, file_type, detections, compliance_status, compliance_percentage):
    """Save detection results to history"""
    history_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": file_name,
        "file_type": file_type,
        "detections": detections,
        "compliance_status": compliance_status,
        "compliance_percentage": compliance_percentage
    }
    st.session_state.detection_history.append(history_entry)

def display_compliance_status(compliance_status, compliance_color, compliance_percentage):
    """Display compliance status with appropriate colors"""
    if compliance_status == "COMPLIANT":
        st.success(f"{compliance_color} **{compliance_status}** ({compliance_percentage:.1f}%)")
    elif compliance_status == "PARTIALLY COMPLIANT":
        st.warning(f"{compliance_color} **{compliance_status}** ({compliance_percentage:.1f}%)")
    else:
        st.error(f"{compliance_color} **{compliance_status}** ({compliance_percentage:.1f}%)")

# Streamlit UI

# Header
st.title("🦺 PPE Detection & Compliance App")
st.write("Upload an image or video to detect PPE and analyze safety compliance. " \
"Note: Sometimes the model names the helemt ot hard hat as No-helmet because of the dataset used as the dataset had some errors and the model is trained on it. " \
"If model fails to detect any ppe item then the epoch on which the model is been trained may be low for that image to detect the ppe items in the image")


# Sidebar for history and settings
with st.sidebar:
    st.header("📊 Detection History")
    if st.session_state.detection_history:
        for i, entry in enumerate(reversed(st.session_state.detection_history[-10:])):  # Show last 10
            with st.expander(f"{entry['file_name']} - {entry['compliance_status']}", expanded=False):
                st.write(f"**Time:** {entry['timestamp']}")
                st.write(f"**Type:** {entry['file_type']}")
                st.write(f"**Compliance:** {entry['compliance_status']} ({entry['compliance_percentage']:.1f}%)")
                st.write(f"**Items Detected:** {len(entry['detections'])}")
    else:
        st.write("No detection history yet.")
    
    if st.button("Clear History"):
        st.session_state.detection_history = []
        st.success("History cleared!")

# Main content
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📋 PPE Requirements")
    req_df = pd.DataFrame([
        {"PPE Item": config["name"], "Required": "Yes" if config["required"] else "No"}
        for config in PPE_REQUIREMENTS.values()
    ])
    st.dataframe(req_df, use_container_width=True)
    
    # Add compliance legend
    st.markdown("**Legend:**")
    st.markdown("🟢 **COMPLIANT** - All required PPE detected")
    st.markdown("🟡 **PARTIALLY COMPLIANT** - Some PPE missing") 
    st.markdown("🔴 **NON-COMPLIANT** - Most/all PPE missing")

with col1:
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Image/Video", 
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        help="Supported formats: JPG, PNG, MP4, AVI, MOV"
    )

    if uploaded_file:
        file_type = uploaded_file.type
        file_name = uploaded_file.name

        # Handle Images
        if file_type.startswith("image"):
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Run YOLOv8 inference
            with st.spinner("🔎 Running detection..."):
                try:
                    results = model.predict(image, conf=0.25)
                    detections = process_detections(results)
                    
                    # Analyze compliance
                    compliance_data, compliance_status, compliance_color, compliance_percentage = analyze_compliance(detections)
                    
                    # Save to history
                    save_to_history(file_name, "Image", detections, compliance_status, compliance_percentage)
                    
                    # Display results
                    st.subheader("🎯 Detection Results")
                    
                    # Show annotated image
                    if results and len(results) > 0:
                        result_img = results[0].plot()
                        st.image(result_img, caption="Detection Result", use_column_width=True)
                    
                    # Compliance Status
                    st.subheader("📊 Compliance Analysis")
                    display_compliance_status(compliance_status, compliance_color, compliance_percentage)
                    
                    # Detection details
                    if detections:
                        st.subheader("🔍 Detection Details")
                        detection_details = []
                        for det in detections:
                            detection_details.append({
                                "Class": det['class'].replace('_', ' ').title(),
                                "Confidence": f"{det['confidence']:.2%}"
                            })
                        
                        details_df = pd.DataFrame(detection_details)
                        st.dataframe(details_df)
                    else:
                        st.warning("No PPE items detected in the image.")
                
                except Exception as e:
                    st.error(f"Error during detection: {e}")

        # Handle Videos
        elif file_type.startswith("video"):
            st.video(uploaded_file)
            
            if st.button("🎬 Process Video"):
                with st.spinner("⚡ Running detection on video..."):
                    try:
                        # Save uploaded video to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            temp_video_path = tmp_file.name
                        
                        # Process video
                        results = model.predict(
                            temp_video_path, 
                            save=True, 
                            project="runs/streamlit_results", 
                            name="ppe_video",
                            conf=0.25
                        )
                        
                        # Analyze first frame for compliance (you can modify this for full video analysis)
                        if results:
                            detections = process_detections([results[0]] if isinstance(results, list) else results)
                            compliance_data, compliance_status, compliance_color, compliance_percentage = analyze_compliance(detections)
                            
                            # Save to history
                            save_to_history(file_name, "Video", detections, compliance_status, compliance_percentage)
                            
                            # Display results
                            st.success("✅ Video processed successfully!")
                            
                            # Compliance Status
                            st.subheader("📊 Compliance Analysis (First Frame)")
                            display_compliance_status(compliance_status, compliance_color, compliance_percentage)
                        
                        # Clean up
                        os.unlink(temp_video_path)
                        
                        st.info("📁 Processed video saved in `runs/streamlit_results/ppe_video/`")
                    
                    except Exception as e:
                        st.error(f"Error processing video: {e}")

# Footer
st.markdown("---")
st.markdown("**Note:** This app analyzes PPE compliance based on detected safety equipment. It's important to Ensure proper PPE usage for workplace safety. " \
" If model fails to detect any ppe item then the epoch on which the model is been trained may be low for that image to detect the ppe items in the image")
