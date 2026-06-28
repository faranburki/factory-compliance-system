"""FastAPI application for the Factory Compliance Dashboard.

Exposes JSON endpoints for policy rules, events, alerts, and export.
Serves static HTML/CSS/JS files from the static/ directory.
"""

from __future__ import annotations

import json
import sys
import shutil
from pathlib import Path
import mimetypes
from typing import Optional

mimetypes.init()
mimetypes.add_type("video/mp4", ".mp4")

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Query, BackgroundTasks, HTTPException

# Ensure src/ is on the path
_SRC_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_ROOT.parent
sys.path.insert(0, str(_SRC_ROOT))
load_dotenv(_PROJECT_ROOT / ".env")

from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from escalation.alert_queue import AlertQueue, create_alert_queue
from policy_parsing.pdf_text_extractor import extract_pdf_text
from policy_parsing.rule_extractor import extract_policy_rules, write_policy_rules_json
from reports.db import fetch_events, initialize_database
from reports.export import export_events_csv, export_events_json
from reports.schema import ReportEvent
from detection.annotator import annotate_video_with_green_borders
from detection.run_detection import run_all_detectors
from escalation.route_event import build_report_event, route_event
from severity.classify_severity import classify_severity
import run_pipeline

# ── Paths ───────────────────────────────────────────────────────────────
DB_PATH = _PROJECT_ROOT / "outputs" / "reports.db"
POLICY_RULES_JSON = _PROJECT_ROOT / "outputs" / "policy_rules.json"
POLICY_PDF = _PROJECT_ROOT / "Compliance_Policy_Manual.pdf"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = _PROJECT_ROOT / "data"
UPLOAD_DIR = _PROJECT_ROOT / "uploads"

# ── App ─────────────────────────────────────────────────────────────────
app = FastAPI(
	title="Factory Compliance Dashboard",
	description="Live monitoring, alert timeline, and historical log for factory safety violations.",
	version="1.0.0",
)

processing_status = {
    "is_processing": False,
    "current_clip": None,
    "status": "idle",
    "progress": 0.0,
    "detail": ""
}

CURRENT_STATE = {
    "status": "idle",
    "clip_id": None,
    "events_this_session": 0,
    "max_severity": None,
    "has_violation": False,
    "detections": []
}

# ── Shared state ────────────────────────────────────────────────────────
POLICY_RULES: list[dict] = []
alert_queue: AlertQueue = create_alert_queue()
recent_alerts: list[dict] = []  # In-memory buffer of recent HIGH/CRITICAL alerts
_last_clip_id: str | None = None


# ── Startup ─────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_policy():
	"""Load policy rules from JSON on startup (Option A from the guide)."""
	global POLICY_RULES, _last_clip_id

	# Clear the database on startup so there's no previous video/alerts
	if DB_PATH.exists():
		try:
			# Also ensure connections are closed before unlinking if possible, 
			# but since it's startup, no connections exist yet.
			DB_PATH.unlink()
			print(f"[startup] Cleared previous database at {DB_PATH}")
		except Exception as exc:
			print(f"[startup] Warning: Could not clear database: {exc}")

	# Initialize the database
	initialize_database(DB_PATH)

	# Load policy rules from cached JSON if available
	if POLICY_RULES_JSON.exists():
		try:
			data = json.loads(POLICY_RULES_JSON.read_text(encoding="utf-8"))
			POLICY_RULES = data.get("rules", [])
			print(f"[startup] Loaded {len(POLICY_RULES)} policy rules from {POLICY_RULES_JSON}")
		except Exception as exc:
			print(f"[startup] Warning: Could not load policy rules: {exc}")
	else:
		print(f"[startup] No policy_rules.json found at {POLICY_RULES_JSON}")

	# Determine last clip_id from existing events (which should be empty now)
	events = fetch_events(DB_PATH)
	if events:
		CURRENT_STATE["clip_id"] = events[0].clip_id
		_last_clip_id = events[0].clip_id
	else:
		CURRENT_STATE["clip_id"] = None
		_last_clip_id = None

# ── API Routes ──────────────────────────────────────────────────────────

@app.get("/api/policy/rules", tags=["Policy"])
def get_policy_rules():
	"""Return the parsed policy rules."""
	return {"rules": POLICY_RULES, "count": len(POLICY_RULES)}


@app.post("/api/policy/parse", tags=["Policy"])
async def parse_policy_upload(file: UploadFile = File(...)):
	"""Upload and parse a policy PDF to extract rules (Option B)."""
	global POLICY_RULES
	contents = await file.read()

	# Write to a temp path for pdfplumber
	temp_path = _PROJECT_ROOT / "outputs" / "uploaded_policy.pdf"
	temp_path.parent.mkdir(parents=True, exist_ok=True)
	temp_path.write_bytes(contents)

	try:
		text = extract_pdf_text(temp_path)
		extraction = extract_policy_rules(text)
		write_policy_rules_json(extraction, POLICY_RULES_JSON)
		POLICY_RULES = [rule.model_dump() for rule in extraction.rules]
		return {"status": "success", "rules": POLICY_RULES, "count": len(POLICY_RULES)}
	except Exception as exc:
		return JSONResponse(
			status_code=500,
			content={"status": "error", "detail": str(exc)},
		)
	finally:
		if temp_path.exists():
			temp_path.unlink()

@app.delete("/api/policy/rules/{index}", tags=["Policy"])
def delete_policy_rule(index: int):
	"""Delete a specific policy rule by index."""
	global POLICY_RULES
	if index < 0 or index >= len(POLICY_RULES):
		raise HTTPException(status_code=404, detail="Rule not found")
	
	deleted_rule = POLICY_RULES.pop(index)
	# Re-write the JSON file
	_save_policy_rules()
	return {"status": "success", "deleted": deleted_rule, "rules": POLICY_RULES, "count": len(POLICY_RULES)}

@app.delete("/api/policy/rules", tags=["Policy"])
def clear_policy_rules():
	"""Clear all policy rules."""
	global POLICY_RULES
	POLICY_RULES.clear()
	_save_policy_rules()
	return {"status": "success", "rules": POLICY_RULES, "count": 0}

def _save_policy_rules():
	"""Helper to save the global POLICY_RULES to the JSON file."""
	extraction = {"rules": POLICY_RULES}
	POLICY_RULES_JSON.parent.mkdir(parents=True, exist_ok=True)
	POLICY_RULES_JSON.write_text(json.dumps(extraction, indent=2), encoding="utf-8")
@app.get("/api/events", tags=["Events"])
def get_events(
	severity: Optional[str] = Query(None, description="Filter by severity tier"),
	behavior_class: Optional[str] = Query(None, description="Filter by behavior class"),
):
	"""Return all violation events from the database with optional filtering."""
	events = fetch_events(DB_PATH, severity=severity, behavior_class=behavior_class)
	return {
		"events": [e.model_dump() for e in events],
		"count": len(events),
	}


# ── Frontend-specific endpoints ─────────────────────────────────────────

@app.get("/api/events/live", tags=["Frontend"])
def get_events_live():
	"""Return the latest clip's detections for the Live Feed view."""
	global _last_clip_id

	# Drain alert queue into recent_alerts buffer
	while not alert_queue.empty():
		alert_event = alert_queue.pop()
		recent_alerts.append(alert_event.model_dump())
	while len(recent_alerts) > 50:
		recent_alerts.pop(0)

	events = fetch_events(DB_PATH)

	# Determine the latest clip from CURRENT_STATE instead of events
	_last_clip_id = CURRENT_STATE.get("clip_id")

	if not _last_clip_id:
		return {
			"status": "idle",
			"clip_id": None,
			"events_this_session": 0,
			"detections": [],
			"has_violation": False,
			"max_severity": None,
		}

	# Get events for the currently playing clip only
	# DB stores original filename (e.g. video.mp4), but clip_id is annotated filename (e.g. video_annotated.mp4)
	base_clip_id = _last_clip_id.replace("_annotated", "") if _last_clip_id else None
	clip_events = [e for e in events if e.clip_id == base_clip_id]

	# Determine alert status
	severities = [e.severity for e in clip_events]
	has_violation = len(clip_events) > 0
	max_severity = None
	for s in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
		if s in severities:
			max_severity = s
			break

	# Inject time_sec into the event dictionaries
	import re
	event_dicts = []
	for e in clip_events:
		e_dict = e.model_dump()
		match = re.search(r'at frame (\d+)', e.event_description)
		if match:
			e_dict['time_sec'] = int(match.group(1)) / 30.0
		else:
			e_dict['time_sec'] = 0.0
		event_dicts.append(e_dict)

	return {
		"status": "alert" if max_severity in ("HIGH", "CRITICAL") else ("violation" if has_violation else "idle"),
		"clip_id": _last_clip_id,
		"events_this_session": len(events),
		"detections": event_dicts,
		"has_violation": has_violation,
		"max_severity": max_severity,
	}


@app.get("/api/events/stream", tags=["Frontend"])
def get_events_stream():
	"""Return the last 200 events in reverse chronological order for the Event Stream view."""
	events = fetch_events(DB_PATH)
	limited = events[:200]
	return {
		"events": [e.model_dump() for e in limited],
		"count": len(limited),
	}


@app.get("/api/events/log", tags=["Frontend"])
def get_events_log(
	start: Optional[str] = Query(None, alias="start", description="Start date (ISO 8601)"),
	end: Optional[str] = Query(None, alias="end", description="End date (ISO 8601)"),
	severity: Optional[str] = Query(None, description="Filter by severity tier"),
	behavior_class: Optional[str] = Query(None, alias="class", description="Filter by behavior class"),
):
	"""Return the filtered full log for the Historical Log view."""
	events = fetch_events(DB_PATH, severity=severity, behavior_class=behavior_class)

	# Apply date range filtering if provided
	if start:
		events = [e for e in events if e.timestamp >= start]
	if end:
		# Add a day to make end inclusive
		events = [e for e in events if e.timestamp <= end + "T23:59:59"]

	return {
		"events": [e.model_dump() for e in events],
		"count": len(events),
	}


# ── Keep original endpoints for backward compatibility ──────────────────

@app.get("/api/live", tags=["Live"])
def get_live_status():
	"""Return the latest events and current alert status for the live feed."""
	events = fetch_events(DB_PATH)
	latest = events[:10] if events else []

	while not alert_queue.empty():
		alert_event = alert_queue.pop()
		recent_alerts.append(alert_event.model_dump())
	while len(recent_alerts) > 50:
		recent_alerts.pop(0)

	has_active_alerts = len(recent_alerts) > 0
	latest_alert = recent_alerts[-1] if recent_alerts else None

	return {
		"status": "alert" if has_active_alerts else "normal",
		"latest_events": [e.model_dump() for e in latest],
		"active_alerts": recent_alerts[-10:],
		"latest_alert": latest_alert,
	}


@app.get("/api/log", tags=["Log"])
def get_historical_log(
	severity: Optional[str] = Query(None, description="Filter by severity tier"),
	behavior_class: Optional[str] = Query(None, description="Filter by behavior class"),
	limit: int = Query(100, description="Max number of events to return"),
):
	"""Return the full historical log, filterable by severity and behavior class."""
	events = fetch_events(DB_PATH, severity=severity, behavior_class=behavior_class)
	limited = events[:limit]
	return {
		"events": [e.model_dump() for e in limited],
		"count": len(limited),
		"total": len(events),
	}


@app.get("/api/export/json", tags=["Export"])
def export_json(
	severity: Optional[str] = Query(None),
	behavior_class: Optional[str] = Query(None),
):
	"""Export all events as a downloadable JSON file."""
	json_str = export_events_json(DB_PATH, severity=severity, behavior_class=behavior_class)
	return JSONResponse(
		content=json.loads(json_str) if json_str.strip() else [],
		headers={"Content-Disposition": "attachment; filename=compliance_events.json"},
	)


@app.get("/api/export/csv", tags=["Export"])
def export_csv(
	severity: Optional[str] = Query(None),
	behavior_class: Optional[str] = Query(None),
):
	"""Export all events as a downloadable CSV file."""
	csv_str = export_events_csv(DB_PATH, severity=severity, behavior_class=behavior_class)
	return PlainTextResponse(
		content=csv_str,
		media_type="text/csv",
		headers={"Content-Disposition": "attachment; filename=compliance_events.csv"},
	)


@app.get("/api/stats", tags=["Stats"])
def get_stats():
	"""Return summary statistics for the dashboard."""
	events = fetch_events(DB_PATH)

	severity_counts = {}
	behavior_counts = {}
	for ev in events:
		severity_counts[ev.severity] = severity_counts.get(ev.severity, 0) + 1
		behavior_counts[ev.behavior_class] = behavior_counts.get(ev.behavior_class, 0) + 1

	return {
		"total_events": len(events),
		"severity_breakdown": severity_counts,
		"behavior_breakdown": behavior_counts,
		"active_alerts": len(recent_alerts),
	}


# ── Root — serve the dashboard HTML ────────────────────────────────────
@app.get("/", tags=["Root"], response_class=HTMLResponse)
def root():
	"""Serve the dashboard HTML page."""
	index_path = STATIC_DIR / "index.html"
	if index_path.exists():
		return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
	return HTMLResponse(content="<h1>Dashboard not found. Place index.html in static/</h1>")


@app.post("/api/video/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
	global processing_status
	if not POLICY_RULES:
		return {"error": "No active policy rules found. Please upload a policy first."}
		
	if processing_status["is_processing"]:
		return {"error": "A video is already being processed. Please wait."}
		
	filename = file.filename
	target_path = DATA_DIR / filename
	
	# Ensure the data directory exists in case it was deleted
	target_path.parent.mkdir(parents=True, exist_ok=True)
	
	# Save the file
	with target_path.open("wb") as buffer:
		shutil.copyfileobj(file.file, buffer)
		
	# Start background task
	processing_status["is_processing"] = True
	processing_status["current_clip"] = filename
	processing_status["status"] = "processing"
	processing_status["progress"] = 0.0
	
	background_tasks.add_task(process_uploaded_video, target_path)
	
	return {"message": "Upload successful, processing started.", "clip_id": filename}

@app.get("/api/video/status")
def get_video_status():
	return processing_status

@app.post("/api/video/reset")
def reset_video_status():
	global processing_status
	processing_status["is_processing"] = False
	processing_status["current_clip"] = None
	processing_status["status"] = "idle"
	processing_status["progress"] = 0.0
	processing_status["detail"] = ""
	return processing_status

def update_progress(p: float):
	global processing_status
	processing_status["progress"] = p

def update_detail(detail: str):
	global processing_status
	processing_status["detail"] = detail

def process_uploaded_video(video_path: Path):
	global processing_status
	try:
		filename = video_path.name
		print(f"Background processing started for {filename}")
		
		# 1. Run detection on raw video FIRST to find violations
		processing_status["status"] = "detecting"
		processing_status["detail"] = "Starting compliance detectors..."
		authorized_detections: list[dict] = []
		detection_results = run_all_detectors(
			video_path,
			skip_vision=False,
			step_callback=update_detail,
			authorized_out=authorized_detections,
		)
		
		# 2. Build frame_index -> list[dict] map for annotation coloring
		# Each dict contains {"severity": str, "box": tuple | None}
		violation_frames: dict[int, list[dict]] = {}
		alert_q = create_alert_queue()
		severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
		# How many frames before/after a detected violation to keep the color active.
		# Detectors sample every ~30 frames, so spread across the full stride window.
		SPREAD = 30
		
		last_alert_frame: dict[str, int] = {}

		for det in detection_results:
			# Determine severity dynamically so we can draw boxes even if we debounce DB entry
			severity = classify_severity(det.behavior_class)

			# 1. Spread violation color across surrounding frames (ALWAYS runs for every detection)
			frame_idx = det.frame_index
			if frame_idx is not None:
				for f in range(max(1, frame_idx - SPREAD), frame_idx + SPREAD + 1):
					if f not in violation_frames:
						violation_frames[f] = []
					# Check if a violation for this person box already exists in this frame
					existing = next((v for v in violation_frames[f] if v["box"] == det.person_box), None)
					if existing is None:
						violation_frames[f].append({"severity": severity, "box": det.person_box, "behavior_class": det.behavior_class})
					elif severity_rank.get(severity, 0) > severity_rank.get(existing["severity"], 0):
						existing["severity"] = severity
						existing["behavior_class"] = det.behavior_class

			# 2. Debounce DB + Alert generation (max 1 per 60 frames / 2 seconds per behavior class)
			if frame_idx is not None:
				last_frame = last_alert_frame.get(det.behavior_class, -999)
				if frame_idx - last_frame < 60:
					continue  # Throttle: skip DB and Alert queue for this frame
				last_alert_frame[det.behavior_class] = frame_idx

			# 3. Create actual report event and escalate
			report_event = build_report_event(
				clip_id=det.clip_id,
				zone=det.zone,
				behavior_class=det.behavior_class,
				policy_rule_ref=det.policy_rule_ref,
				event_description=det.event_description,
				observable_indicator_ref=det.observable_indicator_ref,
			)
			route_event(DB_PATH, report_event, alert_q)
		
		print(f"  Built violation map: {len(violation_frames)} frames with violations")

		# 3. Build authorized_frames map for green vest annotation
		authorized_frames: dict[int, list[dict]] = {}
		for auth in authorized_detections:
			frame_idx = auth["frame_index"]
			if frame_idx is not None:
				for f in range(max(1, frame_idx - SPREAD), frame_idx + SPREAD + 1):
					if f not in authorized_frames:
						authorized_frames[f] = []
					authorized_frames[f].append({"box": auth["person_box"]})
		if authorized_frames:
			print(f"  Built authorized map: {len(authorized_frames)} frames with authorized persons")
		
		# 4. Annotate video with severity-colored bounding boxes
		processing_status["status"] = "annotating"
		processing_status["detail"] = "Drawing severity-colored bounding boxes..."
		annotated_path = DATA_DIR / f"{video_path.stem}_annotated.mp4"
		annotate_video_with_green_borders(
			video_path,
			annotated_path,
			progress_callback=update_progress,
			violation_frames=violation_frames,
			authorized_frames=authorized_frames,
		)
		
		# 4. Update dashboard state so it switches to the annotated video
		CURRENT_STATE["clip_id"] = annotated_path.name
		
		processing_status["status"] = "completed"
		processing_status["progress"] = 1.0
	except Exception as e:
		print(f"Error processing video: {e}")
		import traceback
		traceback.print_exc()
		processing_status["status"] = f"error: {str(e)}"
	finally:
		processing_status["is_processing"] = False

# ── Mount static files AFTER all routes ─────────────────────────────────
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

DATA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")
