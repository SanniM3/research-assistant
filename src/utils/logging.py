"""Logging utilities for the research assistant."""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config.settings import get_settings


def setup_logging(log_file: Optional[str] = None, level: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional path to log file
        level: Optional log level (DEBUG, INFO, WARNING, ERROR)
    """
    settings = get_settings()
    
    log_level = getattr(logging, level or settings.log_level, logging.INFO)
    log_path = log_file or settings.log_file
    
    # Create log directory if needed
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


class ResearchLogger:
    """Structured logger for research workflow events."""
    
    def __init__(self, name: str = "research"):
        self.logger = get_logger(name)
        self.events = []
    
    def log_event(self, event_type: str, agent: str, details: dict = None) -> None:
        """Log a structured event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "agent": agent,
            "details": details or {},
        }
        self.events.append(event)
        
        # Also log to standard logger
        self.logger.info(f"[{agent}] {event_type}: {details}")
    
    def log_query(self, query: str, source: str, results_count: int) -> None:
        """Log a search query execution."""
        self.log_event("query", "retriever", {
            "query": query,
            "source": source,
            "results": results_count,
        })
    
    def log_ingestion(self, paper_id: str, chunks_count: int) -> None:
        """Log paper ingestion."""
        self.log_event("ingestion", "ingestion", {
            "paper_id": paper_id,
            "chunks": chunks_count,
        })
    
    def log_extraction(self, paper_id: str, claims: int, entities: int) -> None:
        """Log knowledge extraction."""
        self.log_event("extraction", "extractor", {
            "paper_id": paper_id,
            "claims": claims,
            "entities": entities,
        })
    
    def log_synthesis(self, section_id: str, claims_used: int) -> None:
        """Log section synthesis."""
        self.log_event("synthesis", "synthesizer", {
            "section_id": section_id,
            "claims_used": claims_used,
        })
    
    def log_issue(self, issue_id: str, severity: str, category: str) -> None:
        """Log issue creation."""
        self.log_event("issue", "verifier", {
            "issue_id": issue_id,
            "severity": severity,
            "category": category,
        })
    
    def get_summary(self) -> dict:
        """Get summary of logged events."""
        summary = {
            "total_events": len(self.events),
            "by_type": {},
            "by_agent": {},
        }
        
        for event in self.events:
            event_type = event["type"]
            agent = event["agent"]
            
            summary["by_type"][event_type] = summary["by_type"].get(event_type, 0) + 1
            summary["by_agent"][agent] = summary["by_agent"].get(agent, 0) + 1
        
        return summary
    
    def export_events(self) -> list:
        """Export all logged events."""
        return self.events.copy()
