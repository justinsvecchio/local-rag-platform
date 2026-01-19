"""Trace writer for persisting retrieval traces."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.models.generation import RetrievalTrace


class TraceWriter:
    """Writer for persisting retrieval traces.

    Traces are useful for debugging, evaluation, and analysis.
    """

    def __init__(
        self,
        output_dir: str | Path = "traces",
        format: str = "jsonl",
        max_files: int = 1000,
    ):
        """Initialize trace writer.

        Args:
            output_dir: Directory to write traces
            format: Output format ('jsonl' or 'json')
            max_files: Maximum trace files to keep
        """
        self.output_dir = Path(output_dir)
        self.format = format
        self.max_files = max_files
        self._traces: list[RetrievalTrace] = []

    def write(self, trace: RetrievalTrace) -> str:
        """Write a trace to storage.

        Args:
            trace: Trace to write

        Returns:
            Path to written file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.format == "jsonl":
            return self._write_jsonl(trace)
        else:
            return self._write_json(trace)

    def _write_jsonl(self, trace: RetrievalTrace) -> str:
        """Write trace to JSONL file."""
        # Use date-based file
        date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = self.output_dir / f"traces_{date_str}.jsonl"

        with open(filepath, "a") as f:
            f.write(trace.model_dump_json() + "\n")

        return str(filepath)

    def _write_json(self, trace: RetrievalTrace) -> str:
        """Write trace to individual JSON file."""
        timestamp = trace.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"trace_{timestamp}_{trace.trace_id[:8]}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            f.write(trace.model_dump_json(indent=2))

        # Cleanup old files if needed
        self._cleanup_old_files()

        return str(filepath)

    def _cleanup_old_files(self) -> None:
        """Remove old trace files if over limit."""
        if self.format != "json":
            return

        files = sorted(
            self.output_dir.glob("trace_*.json"),
            key=lambda p: p.stat().st_mtime,
        )

        while len(files) > self.max_files:
            oldest = files.pop(0)
            oldest.unlink()

    def buffer(self, trace: RetrievalTrace) -> None:
        """Buffer a trace for batch writing.

        Args:
            trace: Trace to buffer
        """
        self._traces.append(trace)

    def flush(self) -> list[str]:
        """Write all buffered traces.

        Returns:
            List of written file paths
        """
        paths = []
        for trace in self._traces:
            path = self.write(trace)
            paths.append(path)
        self._traces = []
        return paths

    def load_traces(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int | None = None,
    ) -> list[RetrievalTrace]:
        """Load traces from storage.

        Args:
            start_date: Filter traces after this date
            end_date: Filter traces before this date
            limit: Maximum traces to return

        Returns:
            List of RetrievalTrace objects
        """
        traces = []

        if self.format == "jsonl":
            traces = self._load_jsonl(start_date, end_date, limit)
        else:
            traces = self._load_json(start_date, end_date, limit)

        return traces

    def _load_jsonl(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
        limit: int | None,
    ) -> list[RetrievalTrace]:
        """Load traces from JSONL files."""
        traces = []

        for filepath in sorted(self.output_dir.glob("traces_*.jsonl")):
            with open(filepath) as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        trace = RetrievalTrace.model_validate(data)

                        # Filter by date
                        if start_date and trace.timestamp < start_date:
                            continue
                        if end_date and trace.timestamp > end_date:
                            continue

                        traces.append(trace)

                        if limit and len(traces) >= limit:
                            return traces
                    except Exception:
                        continue

        return traces

    def _load_json(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
        limit: int | None,
    ) -> list[RetrievalTrace]:
        """Load traces from individual JSON files."""
        traces = []

        for filepath in sorted(self.output_dir.glob("trace_*.json")):
            try:
                with open(filepath) as f:
                    data = json.load(f)

                trace = RetrievalTrace.model_validate(data)

                if start_date and trace.timestamp < start_date:
                    continue
                if end_date and trace.timestamp > end_date:
                    continue

                traces.append(trace)

                if limit and len(traces) >= limit:
                    return traces
            except Exception:
                continue

        return traces
