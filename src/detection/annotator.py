"""Video annotator for drawing green borders on moving people.

Supports severity-aware coloring: if a violation_frames dict is provided,
bounding boxes on violation frames will be colored by severity level.
"""

from pathlib import Path
import cv2
from ultralytics import YOLO

# Minimum bounding box height in pixels to count as a real person.
# This filters out tiny false positives like dustbins, fire extinguishers, etc.
# In a 1080p factory camera, even a distant person is ~70px+ tall.
MIN_PERSON_HEIGHT_PX = 70

# Minimum bounding box area in pixels squared.
# A dustbin might be ~50x25 = 1250px². A real person is 70x30+ = 2100px² minimum.
MIN_PERSON_AREA_PX = 2000

# Minimum aspect ratio (height / width). People are taller than wide.
# A dustbin is roughly square (~1.0). A standing person is ~2.0-4.0.
# A sitting/crouching person can be ~1.5+.
MIN_ASPECT_RATIO = 1.5

# BGR color map for severity tiers
SEVERITY_COLORS = {
    "CRITICAL": (0, 0, 255),       # Red
    "HIGH":     (0, 0, 255),       # Red (combining high and critical to red)
    "MEDIUM":   (0, 165, 255),     # Orange
    "LOW":      (0, 255, 0),       # Green
}
DEFAULT_COLOR = (0, 255, 0)        # Green for no violation

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
) -> None:
    """Read a video, detect people, draw colored borders around them, and save.
    
    Args:
        violation_frames: Optional dict mapping frame_index -> list of dicts:
            {"severity": str, "box": tuple[int, int, int, int] | None}
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if violation_frames is None:
        violation_frames = {}

    print(f"Annotating {input_path.name} -> {output_path.name}...")
    if violation_frames:
        print(f"  Severity coloring enabled for {len(violation_frames)} violation frame(s)")

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
                    # class 0 is 'person' in COCO
                    if int(cls) == 0:
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
                        
                        # Determine box color for THIS specific person
                        best_severity = None
                        best_rank = -1
                        violations_in_frame = violation_frames.get(frame_idx, [])
                        
                        for v in violations_in_frame:
                            v_box = v["box"]
                            v_sev = v["severity"]
                            
                            # If the violation has no specific person box (like panel/forklift),
                            # or if it overlaps strongly with this person, apply the severity
                            if v_box is None or get_iou(current_box, v_box) > 0.4:
                                rank = severity_rank.get(v_sev, 0)
                                if rank > best_rank:
                                    best_rank = rank
                                    best_severity = v_sev

                        color = SEVERITY_COLORS.get(best_severity, DEFAULT_COLOR) if best_severity else DEFAULT_COLOR
                        label = f"Person - {best_severity}" if best_severity else "Person"
                        thickness = 4 if best_severity in ("CRITICAL", "HIGH") else 3

                        # Draw colored border
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        # Label with severity
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Finished annotating {output_path.name}")
