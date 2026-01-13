"""
Multilingual Multi-Agent Academic Research Assistant

A LangGraph-based system that mimics the academic literature review process
and produces survey-style academic reports with strict grounding and traceable citations.
"""
from .graph.workflow import ResearchWorkflow, create_research_workflow
from .models.state import ResearchState
from .config.settings import Settings, get_settings

__version__ = "2.0.0"
__all__ = [
    "ResearchWorkflow",
    "create_research_workflow", 
    "ResearchState",
    "Settings",
    "get_settings",
]
