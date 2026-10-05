"""
Thread-safe per-anchor logging for parallel execution.

Each anchor test gets its own AnchorLogger instance that buffers all output.
After parallel execution completes, logs can be written to per-anchor files.
"""
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AnchorLogEntry:
    """Single log entry with timestamp and optional structured data."""
    timestamp: str
    level: str  # INFO, WARN, ERROR, DEBUG
    message: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class AnchorLogger:
    """
    Thread-safe logger for a single anchor's execution.
    
    Buffers all log entries in memory during parallel execution.
    Provides convenience methods for different log levels.
    Can write accumulated logs to file after execution completes.
    """
    anchor_idx: int
    entries: List[AnchorLogEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def log(self, level: str, message: str, data: Dict[str, Any] = None):
        """Add a log entry with timestamp."""
        with self._lock:
            self.entries.append(AnchorLogEntry(
                timestamp=datetime.now().isoformat(),
                level=level,
                message=message,
                data=data
            ))
    
    def info(self, msg: str, **data):
        """Log an INFO level message."""
        self.log("INFO", msg, data if data else None)
    
    def warn(self, msg: str, **data):
        """Log a WARN level message."""
        self.log("WARN", msg, data if data else None)
    
    def error(self, msg: str, **data):
        """Log an ERROR level message."""
        self.log("ERROR", msg, data if data else None)
    
    def debug(self, msg: str, **data):
        """Log a DEBUG level message."""
        self.log("DEBUG", msg, data if data else None)
    
    def rule(self, title: str = ""):
        """Log a visual separator (equivalent to console.rule)."""
        separator = "=" * 60
        if title:
            self.info(f"{separator} {title} {separator}")
        else:
            self.info(separator)
    
    def panel(self, content: str, title: str = ""):
        """Log content that would be in a panel."""
        self.info(f"[{title}] {content[:500]}..." if len(content) > 500 else f"[{title}] {content}")
    
    def to_file(self, path: str):
        """Write all log entries to a file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"=== Anchor {self.anchor_idx} Execution Log ===\n\n")
            for entry in self.entries:
                line = f"[{entry.timestamp}] {entry.level}: {entry.message}"
                if entry.data:
                    line += f"\n    Data: {entry.data}"
                f.write(line + "\n")
    
    def get_summary(self) -> str:
        """Get a brief summary of logged activity."""
        info_count = sum(1 for e in self.entries if e.level == "INFO")
        warn_count = sum(1 for e in self.entries if e.level == "WARN")
        error_count = sum(1 for e in self.entries if e.level == "ERROR")
        return f"Entries: {len(self.entries)} (INFO:{info_count}, WARN:{warn_count}, ERROR:{error_count})"
