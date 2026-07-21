"""LangGraph workflow definition for the research assistant."""
import time
from typing import Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from ..models.state import ResearchState
from ..utils.logging import get_logger

_progress_logger = get_logger("progress")

# Per-run node visit counter, used for a live heartbeat and loop detection.
_NODE_VISITS: Dict[str, int] = {}
_LOOP_WARN_THRESHOLD = 12


def reset_progress() -> None:
    """Reset the per-run node visit counter (call at the start of each run)."""
    _NODE_VISITS.clear()


def _instrument(name: str, fn):
    """Wrap a node so entry/exit, timing, KB counts and loop warnings are logged.

    This is the main progress-monitoring hook: every node prints a line when it
    starts and finishes, so a long phase (ingestion, extraction) shows a clear
    heartbeat and repeated nodes make iteration loops obvious.
    """
    def wrapped(state: ResearchState):
        _NODE_VISITS[name] = _NODE_VISITS.get(name, 0) + 1
        visits = _NODE_VISITS[name]
        iteration = getattr(state, "iteration", 0)
        phase = getattr(state, "phase", "?")

        _progress_logger.info(
            "\u25b6 %-16s | visit #%d | iter=%s/%s | phase=%s",
            name, visits, iteration, getattr(state, "max_iterations", "?"), phase,
        )
        if visits > _LOOP_WARN_THRESHOLD:
            _progress_logger.warning(
                "  '%s' has now run %d times \u2014 likely stuck in a loop "
                "(revisions=%s, resynths=%s)",
                name, visits, getattr(state, "revision_count", 0),
                getattr(state, "resynth_count", 0),
            )

        t0 = time.time()
        result = fn(state)
        dt = time.time() - t0

        new_phase = result.get("phase", phase) if isinstance(result, dict) else phase
        stats = ""
        try:
            s = state.kb().stats()
            stats = (f"papers={s['papers_reviewed']}/{s['papers_total']} "
                     f"chunks={s['chunks']} claims={s['claims']} entities={s['entities']}")
        except Exception:
            pass
        try:
            from ..agents.base import get_cost
            cost = get_cost().get("usd", 0.0)
        except Exception:
            cost = 0.0

        _progress_logger.info(
            "\u2714 %-16s | %.1fs | \u2192 phase=%s | %s | ~$%.3f",
            name, dt, new_phase, stats, cost,
        )
        return result

    return wrapped
from ..agents.orchestrator import orchestrator_node
from ..agents.planner import planner_node
from ..agents.search_planner import search_planner_node
from ..agents.retriever import retriever_node
from ..agents.triage import triage_node
from ..agents.ingestion import ingestion_node
from ..agents.extractor import extractor_node
from ..agents.kb_curator import kb_curator_node
from ..agents.outline_refiner import outline_refiner_node
from ..agents.synthesizer import synthesizer_node
from ..agents.verifier import verifier_node
from ..agents.gap_scorer import gap_scorer_node
from ..agents.citation_manager import citation_manager_node
from ..agents.reviewer import reviewer_node


def route_after_orchestrator(state: ResearchState) -> str:
    """Route based on current phase after orchestrator.
    
    The orchestrator only runs once (at init) so this almost always
    returns 'planner'.  The other branches exist for resumability.
    """
    phase = state.phase
    
    if phase == "clarify":
        return END
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
    elif phase == "outline_refinement":
        return "outline_refiner"
    elif phase in ("synthesis", "resynthesize"):
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
        return "search_planner"
    else:
        return "outline_refiner"


def route_after_reviewer(state: ResearchState) -> str:
    """Route based on review results."""
    if state.phase == "revision":
        return "search_planner"  # Need more research
    elif state.phase == "resynthesize":
        return "synthesizer"  # Writing quality issues — rewrite sections
    else:
        return "finalizer"


def finalizer_node(state: ResearchState) -> Dict[str, Any]:
    """Finalize the survey, compile output, and persist the knowledge base."""
    from ..agents.base import get_cost

    state.log_action("finalizer", "starting", {})

    kb = state.kb()

    # Compile final report
    final_report = compile_final_report(state, kb)

    # Persist the dynamic KB so this corpus accumulates across runs.
    try:
        kb.persist()
    except Exception as e:
        state.log_action("finalizer", "persist_error", {"error": str(e)})

    state.log_action("finalizer", "completed", {
        "report_length": len(final_report),
        "kb_stats": kb.stats(),
        "estimated_cost_usd": round(get_cost().get("usd", 0.0), 4),
    })
    
    return {
        "final_report": final_report,
        "phase": "complete",
        "estimated_cost_usd": round(get_cost().get("usd", 0.0), 4),
    }


def _strip_leading_heading(text: str) -> str:
    """Remove a leading markdown heading (## ...) from synthesizer output."""
    import re
    stripped = text.lstrip()
    # Match lines like "## Title\n" or "# Title\n" at the very start
    stripped = re.sub(r'^#{1,3}\s+[^\n]*\n+', '', stripped, count=1)
    return stripped


def compile_final_report(state: ResearchState, kb) -> str:
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
    parts.append(generate_abstract(state, kb))
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
        # Strip a leading heading that the synthesizer may have included to
        # avoid duplicate ## headers in the final report.
        content = _strip_leading_heading(content)
        parts.append(content)
        parts.append("")
    
    # References (IEEE style)
    parts.append("## References")
    parts.append("")
    
    # Check if we have the new IEEE-style references
    if "_references_text" in state.bib_entries:
        parts.append(state.bib_entries["_references_text"])
    else:
        # Fallback to basic reference list from reviewed papers
        for i, (paper_id, paper) in enumerate(kb.papers_map().items(), 1):
            authors = ", ".join(paper.authors[:3]) if paper.authors else "Unknown"
            if len(paper.authors) > 3:
                authors += " et al."
            title = paper.title or "Untitled"
            year = paper.year or "n.d."
            venue = f"*{paper.venue}*" if paper.venue else f"*arXiv:{paper.arxiv_id}*" if paper.arxiv_id else ""
            parts.append(f"[{i}] {authors}, \"{title},\" {venue} {year}.")
            parts.append("")
    
    # Appendix: Research Statistics
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## Appendix: Research Statistics")
    parts.append("")
    parts.append(f"- **Papers reviewed**: {kb.reviewed_count()}")
    parts.append(f"- **Claims extracted**: {len(kb.all_claims())}")
    parts.append(f"- **Entities identified**: {len(kb.all_entities())}")
    parts.append(f"- **Relations identified**: {len(kb.all_relations())}")
    parts.append(f"- **Research iterations**: {state.iteration}")
    parts.append(f"- **Estimated LLM cost**: ${state.estimated_cost_usd:.3f}")
    parts.append(f"- **Coverage scores**:")
    parts.append(f"  - Taxonomy: {state.coverage_scores.taxonomy_coverage:.0%}")
    parts.append(f"  - Benchmarks: {state.coverage_scores.benchmark_coverage:.0%}")
    parts.append(f"  - Timeline: {state.coverage_scores.timeline_coverage:.0%}")
    
    return "\n".join(parts)


def generate_abstract(state: ResearchState, kb) -> str:
    """Generate an abstract from the survey content."""
    from ..agents.base import get_llm, create_agent_message
    
    llm = get_llm(role="synthesizer")
    
    # Find intro and conclusion by title, not hardcoded ID
    intro = ""
    conclusion = ""
    for section in state.outline:
        title_lower = section.title.lower()
        if "introduction" in title_lower and not intro:
            intro = state.draft_sections.get(section.section_id, "")[:1000]
        if "conclusion" in title_lower:
            conclusion = state.draft_sections.get(section.section_id, "")[:1000]
    if not intro and state.outline:
        intro = state.draft_sections.get(state.outline[0].section_id, "")[:1000]
    
    methods_covered = len([e for e in kb.all_entities() if e.entity_type.value == "method"])
    lang_note = ""
    if (state.output_language or "en").lower() != "en":
        lang_note = f"\nWRITE THE ABSTRACT IN THIS LANGUAGE: {state.output_language}.\n"

    prompt = f"""Write a concise academic abstract (150-250 words) for this survey.
{lang_note}
TOPIC: {state.topic}
SCOPE: {state.scope}

INTRODUCTION EXCERPT:
{intro}

CONCLUSION EXCERPT:
{conclusion}

STATISTICS:
- Papers reviewed: {kb.reviewed_count()}
- Main methods covered: {methods_covered}

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
    
    # Add all nodes (each wrapped with progress instrumentation)
    workflow.add_node("orchestrator", _instrument("orchestrator", orchestrator_node))
    workflow.add_node("planner", _instrument("planner", planner_node))
    workflow.add_node("search_planner", _instrument("search_planner", search_planner_node))
    workflow.add_node("retriever", _instrument("retriever", retriever_node))
    workflow.add_node("triage", _instrument("triage", triage_node))
    workflow.add_node("ingestion", _instrument("ingestion", ingestion_node))
    workflow.add_node("extractor", _instrument("extractor", extractor_node))
    workflow.add_node("kb_curator", _instrument("kb_curator", kb_curator_node))
    workflow.add_node("outline_refiner", _instrument("outline_refiner", outline_refiner_node))
    workflow.add_node("synthesizer", _instrument("synthesizer", synthesizer_node))
    workflow.add_node("verifier", _instrument("verifier", verifier_node))
    workflow.add_node("gap_scorer", _instrument("gap_scorer", gap_scorer_node))
    workflow.add_node("citation_manager", _instrument("citation_manager", citation_manager_node))
    workflow.add_node("reviewer", _instrument("reviewer", reviewer_node))
    workflow.add_node("finalizer", _instrument("finalizer", finalizer_node))
    
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
            "outline_refiner": "outline_refiner",
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
    workflow.add_edge("kb_curator", "gap_scorer")

    # Gap scorer: loop back for more research, or proceed to outline refinement
    workflow.add_conditional_edges(
        "gap_scorer",
        route_after_gap_scorer,
        {
            "search_planner": "search_planner",
            "outline_refiner": "outline_refiner",
        }
    )

    # After outline is finalized: synthesize, verify, then review
    workflow.add_edge("outline_refiner", "synthesizer")
    workflow.add_edge("synthesizer", "verifier")
    
    # Verifier -> Citation manager -> Reviewer
    workflow.add_edge("verifier", "citation_manager")
    workflow.add_edge("citation_manager", "reviewer")
    
    # Reviewer can loop back for research, rewrite sections, or finalize
    workflow.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {
            "search_planner": "search_planner",
            "synthesizer": "synthesizer",
            "finalizer": "finalizer",
        }
    )
    
    # Finalizer -> END
    workflow.add_edge("finalizer", END)
    
    return workflow


def export_run(state, out_dir: str) -> Dict[str, str]:
    """Write a run's artifacts (report, KB export, slim state, summary) to disk.

    These files are what the inspection scripts and Streamlit app read, and the
    KB export replaces the old in-state knowledge_graph dump.
    """
    import os
    import json

    os.makedirs(out_dir, exist_ok=True)

    # Normalise state to a plain object we can query.
    if isinstance(state, dict):
        report = state.get("final_report", "")
        topic = state.get("topic", "")
        corpus_id = state.get("corpus_id", "")
        iteration = state.get("iteration", 0)
        cost = state.get("estimated_cost_usd", 0.0)
        slim_state = {k: _jsonable(v) for k, v in state.items()
                      if k not in {"messages"}}
    else:
        report = getattr(state, "final_report", "")
        topic = getattr(state, "topic", "")
        corpus_id = getattr(state, "corpus_id", "")
        iteration = getattr(state, "iteration", 0)
        cost = getattr(state, "estimated_cost_usd", 0.0)
        slim_state = json.loads(state.model_dump_json())
        slim_state.pop("messages", None)

    from ..storage.registry import get_knowledge_base
    kb = get_knowledge_base(corpus_id) if corpus_id else None

    paths = {}

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    paths["report"] = report_path

    state_path = os.path.join(out_dir, "final_state.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(slim_state, f, indent=2, default=str)
    paths["final_state"] = state_path

    if kb is not None:
        kg_path = os.path.join(out_dir, "knowledge_graph.json")
        with open(kg_path, "w", encoding="utf-8") as f:
            json.dump(kb.export_graph(), f, indent=2, default=str)
        paths["knowledge_graph"] = kg_path

        summary_path = os.path.join(out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "topic": topic,
                "corpus_id": corpus_id,
                "iterations": iteration,
                "estimated_cost_usd": cost,
                "kb_stats": kb.stats(),
            }, f, indent=2, default=str)
        paths["summary"] = summary_path

    return paths


def _jsonable(value):
    try:
        import json
        json.dumps(value, default=str)
        return value
    except Exception:
        return str(value)


class ResearchWorkflow:
    """Wrapper class for the research workflow."""
    
    def __init__(self, checkpointer=None):
        """Initialize the workflow with optional checkpointer."""
        workflow = create_research_workflow()
        
        if checkpointer is None:
            checkpointer = MemorySaver()
        
        self.graph = workflow.compile(checkpointer=checkpointer)
        # recursion_limit bounds total node steps: high enough for several research
        # iterations + revision/resynth loops, low enough to fail loudly (rather
        # than run forever) if something genuinely loops.
        self.config = {"configurable": {"thread_id": "default"}, "recursion_limit": 100}
    
    def set_thread_id(self, thread_id: str) -> None:
        """Set the thread ID for checkpointing."""
        self.config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    
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
        from ..agents.base import reset_cost
        from ..storage.registry import derive_corpus_id

        reset_cost()
        reset_progress()
        # Initialize state
        initial_state = ResearchState(
            topic=topic,
            user_constraints=constraints,
            max_iterations=max_iterations,
            output_language=output_language,
            corpus_id=derive_corpus_id(topic),
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
        from ..agents.base import reset_cost
        from ..storage.registry import derive_corpus_id

        reset_cost()
        reset_progress()
        initial_state = ResearchState(
            topic=topic,
            user_constraints=kwargs.get("constraints"),
            max_iterations=kwargs.get("max_iterations", 5),
            output_language=kwargs.get("output_language", "en"),
            corpus_id=derive_corpus_id(topic),
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
