# Progress

## Done
- Added a PDF text extractor based on `pdfplumber`.
- Built a Groq-based policy-rule extractor and switched the workspace away from Gemini naming.
- Added a policy-rule schema with the four unsafe behavior classes from the guide.
- Added ground-truth comparison helpers and wrote `outputs/policy_rules.json`.
- Updated `test.py` so the policy parsing path runs as a single local smoke test.
- Noted that `data/0_te1.mp4` is the sample video for detection smoke tests.
- Built a model-free walkway detector path with polygon containment checks.
- Added video utilities for sampling frames and reading metadata from `data/0_te1.mp4`.
- Added a smoke test that verifies the walkway detector on synthetic inside/outside points using the sample video dimensions.
- **Phase 1 — Foundation Fixes (complete)**
- **Phase 2 — Detection Engine: 3 Remaining Detectors (complete)**
- **Phase 3 — Detection Orchestrator (complete)**
- **Phase 4 — Full End-to-End Pipeline (complete)**
- **Phase 5 — Reports Export (complete)**
- **Phase 6 — Dashboard Backend (complete)**
- **Phase 7 — Testing & Verification (complete)**
- **Dashboard Frontend (complete):**
  - Built `index.html` — single-file dashboard with embedded CSS and vanilla JS.
  - Bloomberg-terminal-inspired design: black sidebar (#0f0f0f), white main area (#fafafa), system fonts, severity-only color.
  - Three views: Live Feed (stat strip + camera placeholder + pulsing status bar + detection table), Event Stream (filterable chronological log with severity badges and left borders), Historical Log (full data table + date filters + Export CSV/JSON buttons).
  - Polls `/api/events/live` every 3s, `/api/events/stream` every 5s, `/api/events/log` on filter change.
  - Mock data fallback for standalone testing without backend.
  - Added 3 new FastAPI endpoints to match frontend contract: `/api/events/live`, `/api/events/stream`, `/api/events/log`.
  - Root `/` now serves `index.html` directly.
  - All views verified in browser — all rendering correctly.

## Complete
- All modules wired end-to-end per the Master Build Guide.
- Pipeline runs on `data/0_te1.mp4` without crashing.
- Dashboard shows all three required views.
- 89 events in database, export verified.
