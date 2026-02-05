# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
Bounded span queue with priority-based eviction.

The queue has a fixed capacity.  When full:
- Non-error spans are silently dropped.
- Error spans evict the oldest non-error span to make room.
- If the queue contains only error spans, the incoming error is dropped.

All operations are thread-safe.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

DEFAULT_MAX_QUEUE_SIZE = 2048


class BoundedSpanQueue:
    """Thread-safe bounded queue that preserves error spans over non-error spans.

    :param max_size: Maximum number of spans the queue will hold.
    """

    def __init__(self, max_size: int = DEFAULT_MAX_QUEUE_SIZE) -> None:
        self._max_size = max_size
        self._spans: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """Current number of spans in the queue."""
        with self._lock:
            return len(self._spans)

    def enqueue(self, span: Dict[str, Any]) -> bool:
        """Add a span to the queue.

        If the queue is full:
        - Non-error spans are dropped (returns False).
        - Error spans evict the oldest non-error span to make room.
          If no non-error span exists to evict, the error span is also dropped.

        :param span: The span data dictionary.
        :return: True if the span was enqueued, False if it was dropped.
        """
        with self._lock:
            if len(self._spans) >= self._max_size:
                is_error = span.get("attributes", {}).get("dsgjit.status") == "error"
                if not is_error:
                    return False  # Drop non-error span — queue full

                # Evict oldest non-error span to make room for this error
                for i, existing in enumerate(self._spans):
                    if existing.get("attributes", {}).get("dsgjit.status") != "error":
                        self._spans.pop(i)
                        break
                else:
                    return False  # Queue is all errors; cannot make room

            self._spans.append(span)
            return True

    def drain(self) -> List[Dict[str, Any]]:
        """Atomically remove and return all spans in the queue.

        :return: List of span dicts, or an empty list if the queue was empty.
        """
        with self._lock:
            if not self._spans:
                return []
            spans = self._spans[:]
            self._spans.clear()
            return spans