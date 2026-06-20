"""
Module 1 — Detection Engine | run_detection.py
Orchestrate all four detectors on a video clip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .video_utils import load_video_metadata
from .walkway_detector import (
	default_walkway_polygon,
	detect_walkway_violations_in_video,
)
from .vest_detector import detect_vest_violations_in_video
from .panel_detector import detect_panel_violations_in_video
from .forklift_detector import detect_forklift_violations_in_video

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_RULES_JSON = _PROJECT_ROOT / "outputs" / "policy_rules.json"


def get_policy_rules() -> dict[str, dict]:
	"""
	Load policy rules from JSON dynamically.

	Returns:
		A dictionary mapping behavior_class names to their full rule dictionaries.
	"""
	if not POLICY_RULES_JSON.exists():
		return {}
	try:
		data = json.loads(POLICY_RULES_JSON.read_text(encoding="utf-8"))
		rules_dict = {}
		for rule in data.get("rules", []):
			bc = rule.get("behavior_class", "")
			if bc:
				rules_dict[bc] = rule
		return rules_dict
	except Exception:
		return {}


def _match_policy_rule(keyword: str, rules: dict[str, dict]) -> tuple[str | None, str, list[str]]:
	"""
	Match a detector keyword to the dynamically loaded policy rule.

	Args:
		keyword: A substring indicative of the behavior (e.g. 'walkway', 'vest').
		rules: The dictionary of active policy rules.

	Returns:
		A tuple containing (behavior_class, policy_rule_ref, observable_indicators).
	"""
	for bc, rule in rules.items():
		if keyword.lower() in bc.lower() or keyword.lower() in rule.get("rule_text", "").lower():
			pr = rule.get("policy_rule_ref", "Unknown Section")
			indicators = rule.get("observable_indicators", [])
			return bc, pr, indicators
	return None, "Unknown Section", []


@dataclass
class DetectionResult:
	"""Unified detection result from any of the four detectors."""
	behavior_class: str
	policy_rule_ref: str
	event_description: str
	zone: str
	confidence: float
	frame_index: int
	clip_id: str
	needs_review: bool = False
	person_box: tuple[int, int, int, int] | None = None

	def to_dict(self) -> dict:
		"""
		Convert unified detection result to a standard dictionary.
		"""
		return {
			"behavior_class": self.behavior_class,
			"policy_rule_ref": self.policy_rule_ref,
			"event_description": self.event_description,
			"zone": self.zone,
			"confidence": self.confidence,
			"frame_index": self.frame_index,
			"clip_id": self.clip_id,
			"needs_review": self.needs_review,
		}


def run_all_detectors(
	video_path: str | Path,
	*,
	zone: str = "Production Floor",
	stride: int = 30,
	max_frames: int | None = None,
	vision_stride: int = 60,
	vision_max_frames: int = 5,
	model_name: str = "yolov8m.pt",
	confidence_threshold: float = 0.40,
	skip_vision: bool = False,
	step_callback = None,
) -> list[DetectionResult]:
	"""
	Run all four detectors on a video clip and return unified results.

	Args:
		video_path: Path to the video file.
		zone: Facility zone label.
		stride: Frame sampling stride for walkway and vest detectors.
		max_frames: Max frames to sample for walkway and vest detectors.
		vision_stride: Frame sampling stride for vision-based detectors (panel, forklift).
		vision_max_frames: Max frames for vision-based detectors.
		model_name: YOLO model name/path.
		confidence_threshold: Minimum detection confidence.
		skip_vision: If True, skip Groq vision-based detectors (panel, forklift).
		step_callback: Optional callback function for reporting progress.

	Returns:
		List of DetectionResult objects, one per detected violation.
	"""
	video_path = Path(video_path)
	clip_id = video_path.name
	results: list[DetectionResult] = []
	policy_rules = get_policy_rules()

	# ── 1. Walkway Detector (Class 0) ──────────────────────────────────
	bc_walkway, pr_walkway, ind_walkway = _match_policy_rule("walkway", policy_rules)
	if not bc_walkway:
		print(f"  [1/4] Skipping walkway detector (no matching policy rule)")
		if step_callback: step_callback("[1/4] Skipping walkway detector")
	else:
		print(f"  [1/4] Running walkway detector on {clip_id}...")
		if step_callback: step_callback("[1/4] Running walkway detector...")
		try:
			metadata = load_video_metadata(video_path)
			polygon = default_walkway_polygon(metadata.width, metadata.height)
			walkway_detections = detect_walkway_violations_in_video(
				video_path,
				polygon,
				stride=stride,
				max_frames=max_frames,
				model_name=model_name,
				confidence_threshold=confidence_threshold,
			)
			violation_count = 0
			for det in walkway_detections:
				if det.is_violation:
					violation_count += 1
					results.append(
						DetectionResult(
							behavior_class=bc_walkway,
							policy_rule_ref=pr_walkway,
							event_description=(
								f"Person detected outside walkway boundary at frame {det.frame_index}. "
								f"Foot position {det.foot_point} is outside the green-marked walkway polygon."
							),
							zone=zone,
							confidence=det.confidence,
							frame_index=det.frame_index,
							clip_id=clip_id,
							person_box=det.person_box,
						)
					)
			print(f"        -> {violation_count} walkway violation(s) found in {len(walkway_detections)} detections")
		except Exception as exc:
			print(f"        -> Walkway detector error: {exc}")

	# ── 2. Vest Detector (Class 1) ─────────────────────────────────────
	bc_vest, pr_vest, ind_vest = _match_policy_rule("intervention", policy_rules)
	if not bc_vest:
		print(f"  [2/4] Skipping vest detector (no matching policy rule)")
		if step_callback: step_callback("[2/4] Skipping vest detector")
	else:
		print(f"  [2/4] Running vest detector on {clip_id}...")
		if step_callback: step_callback("[2/4] Running vest detector...")
		try:
			vest_detections = detect_vest_violations_in_video(
				video_path,
				stride=stride,
				max_frames=max_frames,
				model_name=model_name,
				confidence_threshold=confidence_threshold,
			)
			violation_count = 0
			for det in vest_detections:
				if det.is_violation:
					violation_count += 1
					results.append(
						DetectionResult(
							behavior_class=bc_vest,
							policy_rule_ref=pr_vest,
							event_description=(
								f"Person detected wearing {det.vest_color} vest near equipment at frame {det.frame_index}. "
								f"Green vest ratio: {det.green_ratio:.2f}, Red-black ratio: {det.red_ratio:.2f}."
							),
							zone=zone,
							confidence=det.confidence,
							frame_index=det.frame_index,
							clip_id=clip_id,
							person_box=det.person_box,
						)
					)
			print(f"        -> {violation_count} vest violation(s) found in {len(vest_detections)} detections")
		except Exception as exc:
			print(f"        -> Vest detector error: {exc}")

	# ── 3. Panel Detector (Class 2) — Groq vision ─────────────────────
	bc_panel, pr_panel, ind_panel = _match_policy_rule("panel", policy_rules)
	if not bc_panel:
		print(f"  [3/4] Skipping panel detector (no matching policy rule)")
		if step_callback: step_callback("[3/4] Skipping panel detector")
	elif skip_vision:
		print(f"  [3/4] Skipping panel detector (vision disabled)")
		if step_callback: step_callback("[3/4] Skipping panel detector (vision disabled)")
	else:
		print(f"  [3/4] Running panel detector on {clip_id}...")
		if step_callback: step_callback("[3/4] Running panel detector...")
		try:
			panel_detections = detect_panel_violations_in_video(
				video_path,
				stride=vision_stride,
				max_frames=vision_max_frames,
			)
			violation_count = 0
			for det in panel_detections:
				if det.is_violation:
					violation_count += 1
					results.append(
						DetectionResult(
							behavior_class=bc_panel,
							policy_rule_ref=pr_panel,
							event_description=(
								f"Electrical panel cover detected {det.panel_state} at frame {det.frame_index}. "
								f"Groq analysis: {det.description}"
							),
							zone=zone,
							confidence=det.confidence,
							frame_index=det.frame_index,
							clip_id=clip_id,
						)
					)
			print(f"        -> {violation_count} panel violation(s) found in {len(panel_detections)} detections")
		except Exception as exc:
			print(f"        -> Panel detector error: {exc}")

	# ── 4. Forklift Detector (Class 3) — YOLO + Groq vision ───────────
	bc_forklift, pr_forklift, ind_forklift = _match_policy_rule("forklift", policy_rules)
	if not bc_forklift:
		print(f"  [4/4] Skipping forklift detector (no matching policy rule)")
		if step_callback: step_callback("[4/4] Skipping forklift detector")
	elif skip_vision:
		print(f"  [4/4] Skipping forklift detector (vision disabled)")
		if step_callback: step_callback("[4/4] Skipping forklift detector (vision disabled)")
	else:
		print(f"  [4/4] Running forklift detector on {clip_id}...")
		if step_callback: step_callback("[4/4] Running forklift detector...")
		try:
			forklift_detections = detect_forklift_violations_in_video(
				video_path,
				stride=vision_stride,
				max_frames=vision_max_frames,
				model_name=model_name,
				confidence_threshold=confidence_threshold,
			)
			violation_count = 0
			for det in forklift_detections:
				if det.is_violation:
					violation_count += 1
					review_note = " [FLAGGED FOR HUMAN REVIEW]" if det.needs_review else ""
					results.append(
						DetectionResult(
							behavior_class=bc_forklift,
							policy_rule_ref=pr_forklift,
							event_description=(
								f"Forklift detected carrying {det.block_count} blocks at frame {det.frame_index}. "
								f"Threshold is 3 or more blocks.{review_note} "
								f"Groq analysis: {det.description}"
							),
							zone=zone,
							confidence=det.confidence,
							frame_index=det.frame_index,
							clip_id=clip_id,
							needs_review=det.needs_review,
						)
					)
			print(f"        -> {violation_count} forklift violation(s) found in {len(forklift_detections)} detections")
		except Exception as exc:
			print(f"        -> Forklift detector error: {exc}")

	print(f"  Total: {len(results)} violation(s) detected in {clip_id}")
	if step_callback: step_callback(f"Detection complete — {len(results)} violation(s) found")
	return results
