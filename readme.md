# Factory Compliance & Alert Escalation System

An end-to-end automated compliance system that ingests raw factory video, parses a regulatory EHS policy document, detects behavioral violations, classifies them by risk severity, and drives real-time alert workflows via a live operations dashboard.

---

## 1. Setup Instructions

### Prerequisites
- Python 3.9+
- A valid Groq API key for LLM-based policy parsing and vision tasks.

### Installation
1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your environment:
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

3. **Windows Users - Codec Requirement**: 
   To ensure the dashboard can correctly serve processed MP4 videos, the Cisco OpenH264 codec is required. Ensure `openh264-1.8.0-win64.dll` is located in the root of the project directory.

### Running the System

**Step 1: Parse the Policy Document**
Extract the compliance rules dynamically from the provided PDF:
```bash
python extract_rules.py
```
*(This generates `outputs/policy_rules.json`)*

**Step 2: Start the Operations Dashboard**
Run the FastAPI backend and web server:
```bash
uvicorn src.dashboard.main:app --reload --port 8000
```
Open your browser and navigate to: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

**Step 3: Process Videos**
Upload videos directly through the web dashboard, or run the pipeline manually via CLI:
```bash
python src/run_pipeline.py --video data/0_te1.mp4
```

---

## 2. Architecture Description

The system implements a modular, 5-stage pipeline bridging computer vision, natural language understanding, and event-driven workflows:

1. **Module 1: Detection Engine (`src/detection`)**
   - Ingests MP4 video feeds.
   - Runs a hybrid detection stack (local object detection + cloud vision-language models) across frames.
   - Outputs structured violation records containing bounding boxes, frame indices, and matched behavior classes.

2. **Module 2: Severity Categorization Matrix (`src/severity`)**
   - Intercepts raw detections and queries the dynamically extracted policy rules.
   - Classifies each event into `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL` risk tiers based on the matched policy context.

3. **Module 3: Escalation Pipeline (`src/escalation`)**
   - Acts as the workflow router.
   - **LOW/MEDIUM** events are silently logged to the database.
   - **HIGH/CRITICAL** events are logged *and* pushed to an in-memory `AlertQueue` for real-time frontend broadcasting.

4. **Module 4: Automated Report Generation (`src/reports`)**
   - Persists all violations as immutable records in a SQLite database (`outputs/reports.db`).
   - Supports exporting historical data as CSV or JSON.

5. **Module 5: Operations Dashboard (`src/dashboard`)**
   - A FastAPI/Vanilla JS web application.
   - Features a **Live Feed Monitor** with overlaid severity-colored bounding boxes (Green=Safe, Yellow=Medium, Orange=High, Red=Critical).
   - Features a **Live Alert Timeline** powered by server-side polling of the `AlertQueue`.
   - Features a **Historical Log** with filtering and CSV export.

---

## 3. Policy Parsing Approach

The system uses an **LLM-based rule extraction pipeline** (`extract_rules.py`):
1. **Text Extraction**: The `pypdf` library extracts raw text from the unstructured `Compliance_Policy_Manual.pdf`.
2. **Structured Prompting**: The text is fed into `llama-3.1-8b-instant` via the Groq API using a strict, zero-shot structured prompt.
3. **Information Extraction**: The LLM isolates distinct unsafe behavior categories, extracts their manual reference (e.g., Section 3.3.2), quotes the source text, and identifies observable visual indicators (e.g., "red-black vest", "electrical panel").
4. **JSON Schema**: The output is coerced into a rigid JSON format (`outputs/policy_rules.json`) that acts as the absolute ground truth for the downstream Detection and Severity modules.

---

## 4. Severity Mapping Rationale

Risk severity is dynamically mapped based on the hazard context and alerting language found in the policy manual, specifically adhering to the following logic instructed during the LLM parsing phase:

- **MEDIUM Risk**: Behaviors flagged under a standard **"WARNING"** callout in the policy manual (e.g., Safe Walkway Violation, Opened Panel Cover). These indicate a behavioral deviation where hazard is present but not yet acute.
- **CRITICAL Risk**: Behaviors flagged under a **"CRITICAL SAFETY NOTICE"** in the manual (e.g., Unauthorized Intervention, Carrying Overload with Forklift). These denote immediate danger, direct injury risk, or the highest-consequence hazards.

*Note: The severity tiers are assigned dynamically from the JSON rulebook, meaning if the policy document is updated, the pipeline's severity routing automatically adjusts without code changes.*

---

## 5. Model Selection Rationale & Limitations

The detection engine utilizes a **hybrid model approach** to balance speed, cost, and visual reasoning capabilities:

### **YOLOv8m (Ultralytics)**
- **Role**: Detects people (`class 0`) for Walkway Violations and Vest Violations.
- **Rationale**: Local, extremely fast, and highly accurate for standard object detection (people). It easily supports high-FPS tracking and bounding-box coordinate extraction, allowing us to mathematically map foot placement against the walkway polygon.
- **Technique**: Used in conjunction with OpenCV HSV color masking (on torso crops) to detect vest colors.

### **LLaVA / Vision-Language Models (Groq API)**
- **Role**: Detects complex, state-based contextual violations (Opened Panel Covers, Forklift Block Counts).
- **Rationale**: Determining if a panel is "open" or counting specific "standardized blocks" on a forklift is difficult to hard-code using classical CV or standard COCO classes. Zero-shot VLM prompting allows the system to visually reason about complex states without needing a custom fine-tuned dataset.

### **Known Limitations**
1. **API Rate Limiting**: The Vision API (Groq) is subject to rate limits. To mitigate this, the pipeline samples frames heavily (e.g., 1 frame per second) for VLM detection rather than processing every frame.
2. **2D Perspective Distortion**: The walkway polygon is defined in 2D pixel coordinates. Camera lens distortion and perspective depth mean that "foot position" calculations are approximations and may occasionally flag edge cases incorrectly.
3. **Color Constancy**: The OpenCV HSV vest detector is highly sensitive to factory lighting conditions. Shadows on a green vest can occasionally register as black/dark, potentially triggering false positives for Unauthorized Intervention.
