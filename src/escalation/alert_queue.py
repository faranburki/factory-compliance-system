"""In-process alert queue for HIGH/CRITICAL report events."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import SimpleQueue

from reports.schema import ReportEvent


@dataclass
class AlertQueue:
	_queue: SimpleQueue[ReportEvent] = field(default_factory=SimpleQueue)

	def push(self, event: ReportEvent) -> None:
		self._queue.put(event)

	def pop(self) -> ReportEvent:
		return self._queue.get()

	def empty(self) -> bool:
		return self._queue.empty()


def create_alert_queue() -> AlertQueue:
	return AlertQueue()
