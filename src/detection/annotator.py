"""Video annotator for drawing colored borders on detected people.

Supports:
  - Severity-aware coloring: violation frames are colored by severity level.
  - Vest-aware coloring: persons wearing a green vest get a green box
    labeled "Authorized Intervention"; red-black vest violators near
    equipment get a red box labeled "Unauthorized Intervention".
"""

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Minimum bounding box height in pixels to count as a real person.
# This filters out tiny false positives like dustbins, fire extinguishers, etc.
# In a 1080p factory camera, even a distant person is ~70px+ tall.
MIN_PERSON_HEIGHT_PX = 70

# Minimum bounding box area in pixels squared.
# A dustbin might be ~50x25 = 1250px². A real person is 70x30+ = 2100px² minimum.
MIN_PERSON_AREA_PX = 2000

# Minimum aspect ratio (height / width). People are taller than wide, but when
# crouching or bending over machinery, this ratio can easily drop below 1.0.
# Lowered to 0.8 to ensure we don't hide bounding boxes of working personnel.
MIN_ASPECT_RATIO = 0.8

# BGR color map for severity tiers
SEVERITY_COLORS = {
    "CRITICAL": (0, 0, 255),       # Red
    "HIGH":     (0, 0, 255),       # Red (combining high and critical to red)
    "MEDIUM":   (0, 165, 255),     # Orange
    "LOW":      (0, 255, 0),       # Green
}
DEFAULT_COLOR = (0, 255, 0)        # Green for no violation

# ── Vest-specific colors (BGR) ──────────────────────────────────────
AUTHORIZED_COLOR = (0, 255, 0)       # Green box for authorized (green vest)
UNAUTHORIZED_COLOR = (0, 0, 255)     # Red box for unauthorized (red-black vest)

def get_iou(boxA, boxB):
    """Calculate Intersection over Union for two bounding boxes."""
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def annotate_video_with_green_borders(
    input_path: str | Path,
    output_path: str | Path,
    model_name: str = "yolov8m.pt",
    confidence_threshold: float = 0.35,
    progress_callback = None,
    violation_frames: dict[int, list[dict]] | None = None,
    authorized_frames: dict[int, list[dict]] | None = None,
) -> None:
    """Read a video, detect people, draw colored borders around them, and save.
    
    Args:
        violation_frames: Optional dict mapping frame_index -> list of dicts:
            {"severity": str, "box": tuple[int, int, int, int] | None}
        authorized_frames: Optional dict mapping frame_index -> list of dicts:
            {"box": tuple[int, int, int, int]}
            These are green-vest authorized persons that get a green box
            labeled "Authorized Intervention".
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if violation_frames is None:
        violation_frames = {}
    if authorized_frames is None:
        authorized_frames = {}

    print(f"Annotating {input_path.name} -> {output_path.name}...")
    if violation_frames:
        print(f"  Severity coloring enabled for {len(violation_frames)} violation frame(s)")
    if authorized_frames:
        print(f"  Authorized intervention markers enabled for {len(authorized_frames)} frame(s)")

    # Load YOLO model
    model = YOLO(model_name)

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        if progress_callback and total_frames > 0:
            if frame_idx % 5 == 0:
                progress_callback(frame_idx / total_frames)

        # Run detection
        results = model.predict(frame, verbose=False, conf=confidence_threshold)
        
        # Draw bounding boxes
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box, cls in zip(boxes.xyxy, boxes.cls):
                    cls_id = int(cls)
                    if cls_id == 0:
                        x1, y1, x2, y2 = map(int, box[:4])
                        box_h = y2 - y1
                        box_w = x2 - x1

                        # Filter out tiny detections (dustbins, objects)
                        if box_h < MIN_PERSON_HEIGHT_PX:
                            continue

                        # Filter out detections with too small an area
                        if (box_h * box_w) < MIN_PERSON_AREA_PX:
                            continue

                        # Filter out squat/square detections (not person-shaped)
                        if box_w > 0 and (box_h / box_w) < MIN_ASPECT_RATIO:
                            continue

                        current_box = (x1, y1, x2, y2)

                        # ── Check if this person is AUTHORIZED (green vest) ──
                        is_authorized = False
                        auth_entries = authorized_frames.get(frame_idx, [])
                        for a in auth_entries:
                            a_box = a.get("box")
                            if a_box is not None and get_iou(current_box, a_box) > 0.3:
                                is_authorized = True
                                break

                        if is_authorized:
                            # Green box + "Authorized Intervention" label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), AUTHORIZED_COLOR, 3)
                            cv2.putText(
                                frame, "Person",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, AUTHORIZED_COLOR, 2,
                            )
                            continue  # skip violation coloring for this person

                        # ── Check severity-based violation coloring ──────────
                        best_severity = None
                        best_rank = -1
                        violations_in_frame = violation_frames.get(frame_idx, [])
                        
                        for v in violations_in_frame:
                            v_box = v["box"]
                            v_sev = v["severity"]
                            
                            # Only apply the severity if the violation has a bounding box that overlaps with this person
                            if v_box is not None and get_iou(current_box, v_box) > 0.4:
                                rank = severity_rank.get(v_sev, 0)
                                if rank > best_rank:
                                    best_rank = rank
                                    best_severity = v_sev

                        if best_severity and best_severity in ("HIGH", "CRITICAL"):
                            # Red box for vest violations
                            color = UNAUTHORIZED_COLOR
                            label = f"Person - {best_severity}"
                            thickness = 4
                        elif best_severity:
                            color = SEVERITY_COLORS.get(best_severity, DEFAULT_COLOR)
                            label = f"Person - {best_severity}"
                            thickness = 3
                        else:
                            color = DEFAULT_COLOR
                            label = "Person"
                            thickness = 3

                        # Draw colored border
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        # Label with severity
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    # ── Forklift Annotation Logic ──
                    elif cls_id in (2, 5, 7):
                        x1, y1, x2, y2 = map(int, box[:4])
                        best_severity = None
                        best_rank = -1
                        violations_in_frame = violation_frames.get(frame_idx, [])
                        
                        for v in violations_in_frame:
                            if v.get("behavior_class", "") == "Carrying Overload with Forklift":
                                v_sev = v["severity"]
                                rank = severity_rank.get(v_sev, 0)
                                if rank > best_rank:
                                    best_rank = rank
                                    best_severity = v_sev
                        
                        if best_severity:
                            color = UNAUTHORIZED_COLOR if best_severity in ("HIGH", "CRITICAL") else SEVERITY_COLORS.get(best_severity, DEFAULT_COLOR)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                            cv2.putText(frame, f"Forklift - {best_severity}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Finished annotating {output_path.name}")
