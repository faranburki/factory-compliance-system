"""
Module 5 — Live Dashboard | main.py
FastAPI server providing a real-time web interface for compliance monitoring.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

try:
	from fastapi import FastAPI
	from fastapi.responses import HTMLResponse
	from fastapi.staticfiles import StaticFiles
	from sse_starlette.sse import EventSourceResponse
except ImportError:  # pragma: no cover
	FastAPI = None
	HTMLResponse = None
	StaticFiles = None
	EventSourceResponse = None

from ..escalation.alert_queue import pop_all_alerts
from ..reports.db import ReportDB

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = _PROJECT_ROOT / "outputs" / "reports.db"

# Ensure FastAPI is installed
if FastAPI is None:
	raise RuntimeError("FastAPI, uvicorn, and sse-starlette must be installed to run the dashboard.")

app = FastAPI(title="Factory Compliance Dashboard")

# Mount the static directory to serve HTML/JS/CSS assets
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
	# For development convenience, ensure the directory exists
	static_dir.mkdir(parents=True, exist_ok=True)
	(static_dir / "index.html").write_text("<h1>Dashboard Static Files Not Found</h1>")

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
	"""
	Serve the main dashboard HTML page.
	"""
	index_path = static_dir / "index.html"
	return index_path.read_text(encoding="utf-8")


@app.get("/api/historical")
async def get_historical_events():
	"""
	API endpoint to retrieve the 100 most recent compliance events from SQLite.

	Returns:
		JSON array of event dictionaries.
	"""
	db = ReportDB(DB_PATH)
	return db.get_recent_events(limit=100)


async def alert_event_generator():
	"""
	Server-Sent Events (SSE) generator for real-time high-severity alerts.

	Yields:
		JSON formatted string payloads as they appear in the AlertQueue.
	"""
	while True:
		# Drain all pending alerts from the queue
		alerts = pop_all_alerts()
		for alert in alerts:
			# Format specifically for SSE
			yield {
				"event": "alert",
				"data": alert,
			}
		
		# Prevent tight looping; check for new alerts twice a second
		await asyncio.sleep(0.5)


@app.get("/api/stream")
async def sse_stream():
	"""
	API endpoint establishing an SSE connection to stream real-time alerts.

	Returns:
		An EventSourceResponse streaming the alert_event_generator.
	"""
	return EventSourceResponse(alert_event_generator())


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8000)
