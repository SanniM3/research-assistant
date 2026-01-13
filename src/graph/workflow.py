"""LangGraph workflow definition for the research assistant."""
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from ..models.state import ResearchState
from ..agents.orchestrator import orchestrator_node
from ..agents.planner import planner_node
from ..agents.search_planner import search_planner_node
from ..agents.retriever import retriever_node
from ..agents.triage import triage_node
from ..agents.ingestion import ingestion_node
from ..agents.extractor import extractor_node
from ..agents.kb_curator import kb_curator_node
from ..agents.synthesizer import synthesizer_node
from ..agents.verifier import verifier_node
from ..agents.gap_scorer import gap_scorer_node
from ..agents.citation_manager import citation_manager_node
from ..agents.reviewer import reviewer_node


def route_after_orchestrator(state: ResearchState) -> str:
    """Route based on current phase after orchestrator."""
    phase = state.phase
    
    if phase == "clarify":
        return END  # Need user input
    elif phase == "planning":
        return "planner"
    elif phase == "search_planning":
        return "search_planner"
    elif phase == "retrieval":
        return "retriever"
    elif phase == "triage":
        return "triage"
    elif phase == "ingestion":
        return "ingestion"
    elif phase == "extraction":
        return "extractor"
    elif phase == "kb_update":
        return "kb_curator"
    elif phase == "synthesis":
        return "synthesizer"
    elif phase == "verification":
        return "verifier"
    elif phase == "gap_scoring":
        return "gap_scorer"
    elif phase == "review":
        return "reviewer"
    elif phase == "finalize":
        return "finalizer"
    else:
        return "planner"


def route_after_gap_scorer(state: ResearchState) -> str:
    """Route based on gap scoring results."""
    if state.phase == "search_planning":
        return "search_planner"  # Continue research loop
    else:
        return "reviewer"  # Proceed to review


def route_after_reviewer(state: ResearchState) -> str:
    """Route based on review results."""
    if state.phase == "revision":
        return "search_planner"  # Need more research
    else:
        return "finalizer"


def finalizer_node(state: ResearchState) -> Dict[str, Any]:
    """Finalize the survey and compile output."""
    state.log_action("finalizer", "starting", {})
    
    # Compile final report
    final_report = compile_final_report(state)
    
    state.log_action("finalizer", "completed", {
        "report_length": len(final_report)
    })
    
    return {
        "final_report": final_report,
        "phase": "complete",
    }


def compile_final_report(state: ResearchState) -> str:
    """Compile the final survey report."""
    parts = []
    
    # Title and metadata
    parts.append(f"# {state.topic}")
    parts.append("")
    parts.append("---")
    parts.append("")
    
    # Abstract
    parts.append("## Abstract")
    parts.append("")
    parts.append(generate_abstract(state))
    parts.append("")
    
    # Table of Contents
    parts.append("## Table of Contents")
    parts.append("")
    for i, section in enumerate(sorted(state.outline, key=lambda s: s.order), 1):
        parts.append(f"{i}. [{section.title}](#{section.section_id})")
    parts.append("")
    
    # Sections
    for section in sorted(state.outline, key=lambda s: s.order):
        parts.append(f"## {section.title}")
        parts.append("")
        content = state.draft_sections.get(section.section_id, "*Section content not available.*")
        parts.append(content)
        parts.append("")
    
    # References
    parts.append("## References")
    parts.append("")
    parts.append("```bibtex")
    for citekey, bibtex in state.bib_entries.items():
        parts.append(bibtex)
        parts.append("")
    parts.append("```")
    
    # Appendix: Research Statistics
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## Appendix: Research Statistics")
    parts.append("")
    parts.append(f"- **Papers reviewed**: {len(state.papers_ingested)}")
    parts.append(f"- **Claims extracted**: {len(state.claims)}")
    parts.append(f"- **Entities identified**: {len(state.entities)}")
    parts.append(f"- **Research iterations**: {state.iteration}")
    parts.append(f"- **Coverage scores**:")
    parts.append(f"  - Taxonomy: {state.coverage_scores.taxonomy_coverage:.0%}")
    parts.append(f"  - Benchmarks: {state.coverage_scores.benchmark_coverage:.0%}")
    parts.append(f"  - Timeline: {state.coverage_scores.timeline_coverage:.0%}")
    
    return "\n".join(parts)


def generate_abstract(state: ResearchState) -> str:
    """Generate an abstract from the survey content."""
    from ..agents.base import get_llm, create_agent_message
    
    llm = get_llm()
    
    # Get intro and conclusion content
    intro = state.draft_sections.get("sec_1", "")[:1000]
    conclusion = ""
    for section in state.outline:
        if "conclusion" in section.title.lower():
            conclusion = state.draft_sections.get(section.section_id, "")[:1000]
            break
    
    prompt = f"""Write a concise academic abstract (150-250 words) for this survey.

TOPIC: {state.topic}
SCOPE: {state.scope}

INTRODUCTION EXCERPT:
{intro}

CONCLUSION EXCERPT:
{conclusion}

STATISTICS:
- Papers reviewed: {len(state.papers_ingested)}
- Main methods covered: {len([e for e in state.entities.values() if e.entity_type.value == "method"])}

Write a clear, informative abstract that:
1. States the survey's purpose
2. Summarizes the scope and methodology
3. Highlights key findings
4. Notes contributions

Output ONLY the abstract text, no formatting."""

    messages = create_agent_message("synthesizer", prompt)
    response = llm.invoke(messages)
    
    return response.content


def create_research_workflow() -> StateGraph:
    """Create the full research workflow graph."""
    
    # Create the graph
    workflow = StateGraph(ResearchState)
    
    # Add all nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("search_planner", search_planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("triage", triage_node)
    workflow.add_node("ingestion", ingestion_node)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("kb_curator", kb_curator_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("gap_scorer", gap_scorer_node)
    workflow.add_node("citation_manager", citation_manager_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("finalizer", finalizer_node)
    
    # Add edges
    
    # Start -> Orchestrator
    workflow.add_edge(START, "orchestrator")
    
    # Orchestrator routes to appropriate phase
    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "planner": "planner",
            "search_planner": "search_planner",
            "retriever": "retriever",
            "triage": "triage",
            "ingestion": "ingestion",
            "extractor": "extractor",
            "kb_curator": "kb_curator",
            "synthesizer": "synthesizer",
            "verifier": "verifier",
            "gap_scorer": "gap_scorer",
            "reviewer": "reviewer",
            "finalizer": "finalizer",
            END: END,
        }
    )
    
    # Linear flow through research phases
    workflow.add_edge("planner", "search_planner")
    workflow.add_edge("search_planner", "retriever")
    workflow.add_edge("retriever", "triage")
    workflow.add_edge("triage", "ingestion")
    workflow.add_edge("ingestion", "extractor")
    workflow.add_edge("extractor", "kb_curator")
    workflow.add_edge("kb_curator", "synthesizer")
    workflow.add_edge("synthesizer", "verifier")
    workflow.add_edge("verifier", "gap_scorer")
    
    # Gap scorer can loop back or proceed
    workflow.add_conditional_edges(
        "gap_scorer",
        route_after_gap_scorer,
        {
            "search_planner": "search_planner",
            "reviewer": "citation_manager",
        }
    )
    
    # Citation manager -> Reviewer
    workflow.add_edge("citation_manager", "reviewer")
    
    # Reviewer can loop back or finalize
    workflow.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {
            "search_planner": "search_planner",
            "finalizer": "finalizer",
        }
    )
    
    # Finalizer -> END
    workflow.add_edge("finalizer", END)
    
    return workflow


class ResearchWorkflow:
    """Wrapper class for the research workflow."""
    
    def __init__(self, checkpointer=None):
        """Initialize the workflow with optional checkpointer."""
        workflow = create_research_workflow()
        
        if checkpointer is None:
            checkpointer = MemorySaver()
        
        self.graph = workflow.compile(checkpointer=checkpointer)
        self.config = {"configurable": {"thread_id": "default"}}
    
    def set_thread_id(self, thread_id: str) -> None:
        """Set the thread ID for checkpointing."""
        self.config = {"configurable": {"thread_id": thread_id}}
    
    def run(self, topic: str, constraints: str = None, 
            max_iterations: int = 5, output_language: str = "en") -> ResearchState:
        """
        Run the full research workflow.
        
        Args:
            topic: Research topic
            constraints: Optional user constraints
            max_iterations: Maximum research iterations
            output_language: Output language code
        
        Returns:
            Final ResearchState with completed survey
        """
        # Initialize state
        initial_state = ResearchState(
            topic=topic,
            user_constraints=constraints,
            max_iterations=max_iterations,
            output_language=output_language,
            phase="init",
        )
        
        # Run the workflow
        final_state = self.graph.invoke(initial_state, config=self.config)
        
        return final_state
    
    def run_with_streaming(self, topic: str, **kwargs):
        """
        Run the workflow with streaming updates.
        
        Yields state updates as the workflow progresses.
        """
        initial_state = ResearchState(
            topic=topic,
            user_constraints=kwargs.get("constraints"),
            max_iterations=kwargs.get("max_iterations", 5),
            output_language=kwargs.get("output_language", "en"),
            phase="init",
        )
        
        for state in self.graph.stream(initial_state, config=self.config):
            yield state
    
    def resume(self, state: ResearchState) -> ResearchState:
        """Resume workflow from a given state."""
        return self.graph.invoke(state, config=self.config)
    
    def get_state(self) -> ResearchState:
        """Get current workflow state."""
        return self.graph.get_state(config=self.config)
