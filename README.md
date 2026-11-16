# PPE Detection & Compliance Monitoring System

An end-to-end computer vision web application built with **Python**, **Streamlit**, **YOLOv8 (Ultralytics)**, and **PyTorch**. Designed for industrial and construction safety management, this platform automates the detection of Personal Protective Equipment (PPE) across images and video feeds, translating raw object detections into real-time, actionable safety compliance scores.

**Live Web Application:** [PPE Detection & Compliance App by Pravar Golecha](https://ppe-detection-model-bypravargolecha.streamlit.app/)  

---

## Brief Overview
I designed and developed the **PPE Detection & Compliance App** to modernize workplace safety inspections using deep learning. By combining a custom-trained **YOLOv8** object detection architecture (`runs/detect/train/weights/best.pt`) with a responsive **Streamlit** interface, the system automatically locates personnel and essential safety gear—including **Hard Hats (Helmets)**, **Safety Vests**, **Protective Gloves**, and **Safety Boots**. Beyond simple bounding-box visualization, the application evaluates detected gear against mandatory site safety rules (`PPE_REQUIREMENTS`) to generate instant compliance percentages, color-coded safety alerts, confidence metrics, and session-level audit logs.

---

## Problem Statement and Brief About It
Ensuring strict adherence to safety protocols in high-risk environments such as construction sites, manufacturing plants, warehouses, and chemical facilities is critical to preventing workplace injuries and fatalities. However, conventional safety monitoring suffers from three major operational bottlenecks:

1. **Limitations of Manual Supervision:** Safety officers cannot continuously monitor every worker across expansive, multi-zone worksites simultaneously. Human fatigue, blind spots, and shift transitions inevitably lead to missed safety violations.
2. **Reactive Rather Than Proactive Enforcement:** Traditional safety audits are periodic rather than continuous. Violations—such as workers entering hazardous zones without hard hats, high-visibility vests, gloves, or steel-toed boots—are frequently identified only after an incident or near-miss occurs.
3. **Lack of Quantifiable Compliance Metrics:** Visual walkthroughs rarely produce standardized, structured compliance records, making it difficult for site managers to track safety trends, audit specific media footage, or verify compliance percentages objectively.

---

## Solution
To solve these challenges, I built an automated, browser-accessible **AI Safety Compliance Engine** powered by YOLOv8 and Streamlit. 

Instead of requiring specialized hardware installations or manual frame-by-frame inspection, safety supervisors can upload site photographs or surveillance video clips (`JPG`, `JPEG`, `PNG`, `MP4`, `AVI`, `MOV`) directly through the web interface. The backend runs real-time neural network inference at a tuned confidence threshold (`conf=0.25`), normalizes detected object classes against standard safety taxonomy, cross-references them with mandatory site rules, and immediately outputs an annotated visual report alongside a quantitative compliance classification:
* 🟢 **COMPLIANT (100%):** All mandatory PPE items and personnel requirements are satisfied.
* 🟡 **PARTIALLY COMPLIANT (50% – 99.9%):** Core presence is detected, but one or more mandatory protective items are missing.
* 🔴 **NON-COMPLIANT (< 50%):** Critical protective equipment is largely or completely absent.

---

##  More About the Solution and Its Features

### 1. System Architecture & Inference Pipeline
The application follows a modular, four-stage processing pipeline inside `ppe_detection.py`:
1. **Media Ingestion & Validation:** Accepts high-resolution images and video files up to **200MB** per upload. Videos are securely buffered into temporary system storage (`tempfile.NamedTemporaryFile`) for frame-level inference and automatically unlinked post-processing to maintain clean disk usage.
2. **Cached Neural Network Engine (`@st.cache_resource`):** Loads custom YOLOv8 weights (`runs/detect/train/weights/best.pt`) into memory once per session for low-latency inference, featuring an automatic fallback mechanism to `yolov8n.pt` to guarantee uninterrupted availability.
3. **Semantic Class Normalization Layer:** Real-world object detection datasets often output synonymous labels. My `analyze_compliance()` engine maps diverse label variants into five canonical safety categories:
   * `helmet` ← `['helmet', 'hard_hat', 'hardhat', 'safety_helmet', 'head_protection']`
   * `safety_vest` ← `['safety_vest', 'vest', 'high_vis', 'high_visibility_vest', 'reflective_vest']`
   * `gloves` ← `['gloves', 'glove', 'hand_protection']`
   * `boots` ← `['boots', 'boot', 'safety_boots', 'work_boots', 'steel_toe']`
   * `person` ← `['person', 'people', 'worker', 'human']`
4. **Compliance Scoring & Audit Logging:** Calculates exact detection counts and compliance ratios (`detected_required / required_items * 100`), renders annotated bounding-box media via `results[0].plot()`, and logs every inspection into `st.session_state.detection_history`.

### 2. Core Features
* ** Dual-Mode Media Analysis (Image & Video):** Seamlessly processes static site photography with instant bounding-box overlays and confidence score tables, as well as video footage with automated output saving to `runs/streamlit_results/ppe_video/`.
* ** Configurable PPE Ruleset (`PPE_REQUIREMENTS`):** Displays an active requirements matrix right on the dashboard so inspectors have clear visibility into which gear is mandatory (`Safety Vest`, `Person`, `Gloves`, `Safety Boots`, `Hard Hat`).
* ** Granular Detection Breakdown:** Generates a structured Pandas DataFrame for every analyzed image displaying each detected class name and its exact model confidence percentage (e.g., `94.20%`).
* ** Interactive Session Audit Sidebar:** Automatically tracks and displays the last **10 inspections** in expandable sidebar cards containing timestamps, media types, compliance percentages, and total detected object counts, complete with a one-click `Clear History` control.

### 3. Advantages & Innovativeness
* **Bridging Computer Vision with Regulatory Logic:** Raw object detection models only output coordinates and class IDs. By wrapping YOLOv8 in a rule-based compliance engine with semantic class normalization, this project converts raw ML predictions into business-ready safety verdicts.
* **Zero-Footprint Cloud Deployment:** Hosted directly on Streamlit Cloud, any site engineer or safety auditor can perform AI-backed PPE checks from a laptop or smartphone browser with zero local GPU or Python setup.
* **Production-Hardened Runtime Compatibility:** Engineered with explicit PyTorch 2.6+ safe-global serialization handling and Streamlit watcher isolation, ensuring reliable execution across modern deep learning environments.

---

##  Future Scope and Currently in Development
* **Full-Video Temporal Compliance Tracking:** Expanding video inference from initial-frame compliance evaluation to continuous frame-by-frame tracking, calculating time-weighted compliance percentages and flagging exact timestamps where violations occur.
* **Live CCTV / RTSP & WebRTC Streaming:** Integrating live camera stream ingestion so safety control rooms can monitor active construction zones and factory floors in real time without manual file uploads.
* **Person-to-PPE Spatial Association:** Implementing bounding-box IoU (Intersection over Union) and spatial containment checks to verify that each detected `person` individually wears a `helmet`, `safety_vest`, `gloves`, and `boots` rather than checking scene-level presence alone.
* **Expanded Multi-Hazard Detection & High-Epoch Retraining:** Scaling training epochs and expanding dataset annotations to cover specialized gear such as safety goggles, ear defenders, face shields, and fall-arrest harnesses.
* **Automated Alerting & Exportable PDF/CSV Audit Reports:** Adding webhook notifications (Email/Slack/SMS) for `NON-COMPLIANT` detections and downloadable compliance reports for OSHA/ISO safety documentation.

---

##  Problems Faced During the Project
Building and deploying a PyTorch + YOLOv8 application on a cloud web environment introduced several technical challenges that I systematically diagnosed and resolved:

1. **PyTorch 2.6 `weights_only` Unpickling Restrictions:**
   * *Challenge:* Upgrading to PyTorch 2.6 enforced stricter `weights_only=True` security defaults during `torch.load()`, blocking custom Ultralytics YOLOv8 checkpoint deserialization.
   * *Resolution:* Explicitly registered core Ultralytics architecture classes (`DetectionModel`, `C2f`, `SPPF`, `Conv`, and `Detect`) via `torch.serialization.add_safe_globals()` prior to model initialization.
2. **Streamlit File-Watcher & PyTorch Runtime Conflicts:**
   * *Challenge:* Streamlit's hot-reload file watcher inspected PyTorch's internal C++ module paths during startup, causing runtime inspection errors.
   * *Resolution:* Configured `os.environ["STREAMLIT_WATCHER_IGNORE"] = "tornado,torch"` at the very top of `ppe_detection.py` before importing Streamlit.
3. **Python Standard Library Module Collisions:**
   * *Challenge:* Local script naming or residual environment modules named `code` interfered with internal Python/Ultralytics imports.
   * *Resolution:* Added a pre-import environment check on `sys.modules['code']` to purge conflicting module references cleanly before loading `ultralytics`.
4. **Dataset Label Inconsistencies & Epoch Sensitivity:**
   * *Challenge:* Variations in training dataset annotations (e.g., `helmet` vs. `hard_hat` or `No-helmet` label noise) and lower training epoch sensitivity occasionally impacted detection consistency on complex angles.
   * *Resolution:* Built the dictionary-based `class_mapping` normalization layer to unify synonymous predictions and calibrated the inference confidence threshold (`conf=0.25`) to balance recall and precision across varied lighting conditions.

---

##  Installation & Local Setup

### Prerequisites
* Python 3.9 or higher
* `pip` package manager

### Step-by-Step Guide
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Pravargolecha2011229/PPE-detection-model.git](https://github.com/Pravargolecha2011229/PPE-detection-model.git)
   cd PPE-detection-model
