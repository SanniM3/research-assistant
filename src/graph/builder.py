from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .types import ResearchState, ResearchGraphState
from .nodes import (
    generate_question,
    search_web,
    search_wikipedia,
    generate_answer,
    save_findings,
    continue_search,
    create_researchers,
    human_feedback,
    write_report,
    run_all_researchers,
)

def build_research_cycle() -> StateGraph:
    """Builds the subgraph that models a single researcher's literature review process."""
    research_cycle = StateGraph(ResearchState)

    research_cycle.add_node("internal_lit_review", generate_question)
    research_cycle.add_node("search_web", search_web)
    research_cycle.add_node("search_wikipedia", search_wikipedia)
    research_cycle.add_node("external_lit_review", generate_answer)
    research_cycle.add_node("save_findings", save_findings)

    research_cycle.add_edge(START, "internal_lit_review")
    research_cycle.add_edge("internal_lit_review", "search_web")
    research_cycle.add_edge("internal_lit_review", "search_wikipedia")
    research_cycle.add_edge("search_web", "external_lit_review")
    research_cycle.add_edge("search_wikipedia", "external_lit_review")
    research_cycle.add_conditional_edges("external_lit_review", continue_search, [
        "internal_lit_review",
        "save_findings",
    ])
    research_cycle.add_edge("save_findings", END)

    return research_cycle.compile(checkpointer=MemorySaver())

def build_full_graph():
    """Builds the full graph that creates researchers, conducts their research, and writes a report."""
    research_cycle = build_research_cycle()

    builder_final = StateGraph(ResearchGraphState)
    builder_final.add_node("create_researchers", create_researchers)
    builder_final.add_node("human_feedback", human_feedback)
    builder_final.add_node("conduct_research", research_cycle)
    builder_final.add_node("write_report", write_report)

    builder_final.add_edge(START, "create_researchers")
    builder_final.add_edge("create_researchers", "human_feedback")
    builder_final.add_conditional_edges("human_feedback", run_all_researchers, [
        "create_researchers",
        "conduct_research",
    ])
    builder_final.add_edge("conduct_research", "write_report")
    builder_final.add_edge("write_report", END)

    return builder_final.compile(
        interrupt_before=["human_feedback"],
        checkpointer=MemorySaver()
    )

graph = build_full_graph()
