"""Search planner agent - generates search queries."""
import json
from typing import Dict, Any, List
from langchain_core.messages import AIMessage

from ..models.state import ResearchState
from ..models.issue import IssueCategory
from .base import get_llm, create_agent_message, parse_llm_json


def search_planner_node(state: ResearchState) -> Dict[str, Any]:
    """
    Search Planner node - generates targeted search queries.
    
    Responsibilities:
    - Generate arXiv queries for academic papers
    - Generate web queries for surveys, datasets, benchmarks
    - Create multilingual variants if needed
    - Plan gap-filling queries based on issues
    """
    llm = get_llm()
    
    state.log_action("search_planner", "generating_queries", {"iteration": state.iteration})
    
    # Gather context for query generation
    existing_queries = [q.query_text for q in state.queries_run]
    open_issues = state.get_open_issues()
    gap_issues = [i for i in open_issues if i.category in [
        IssueCategory.THIN_COVERAGE,
        IssueCategory.TAXONOMY_GAP,
        IssueCategory.BENCHMARK_GAP,
        IssueCategory.MISSING_SEMINAL,
        IssueCategory.MISSING_RECENT,
        IssueCategory.NEEDS_FOLLOW_UP,
    ]]
    
    # Build context about what we already have
    ingested_titles = [p.title for p in state.papers_ingested.values()][:20]
    
    prompt = f"""Generate search queries for academic research on:

TOPIC: {state.topic}
SCOPE: {state.scope}

RESEARCH QUESTIONS (open/partially answered):
{json.dumps([{"id": q.question_id, "text": q.text, "status": q.status.value} for q in state.get_open_questions()], indent=2)}

OUTLINE SECTIONS:
{json.dumps([{"title": s.title, "description": s.description} for s in state.outline], indent=2)}

ITERATION: {state.iteration}

QUERIES ALREADY RUN:
{json.dumps(existing_queries[-10:], indent=2) if existing_queries else "None yet"}

PAPERS ALREADY INGESTED (sample titles):
{json.dumps(ingested_titles, indent=2) if ingested_titles else "None yet"}

GAPS TO ADDRESS:
{json.dumps([{"category": i.category.value, "description": i.description, "suggested": i.suggested_queries} for i in gap_issues], indent=2) if gap_issues else "None identified"}

Generate queries in the following JSON format:
{{
    "arxiv_queries": [
        {{
            "query": "search query text",
            "categories": ["cs.CL", "cs.AI"],
            "rationale": "why this query",
            "target_section": "sec_id or null"
        }}
    ],
    "web_queries": [
        {{
            "query": "search query text",
            "search_type": "academic|dataset|benchmark|general",
            "rationale": "why this query",
            "target_section": "sec_id or null"
        }}
    ]
}}

Guidelines:
1. Generate 3-5 arXiv queries covering different aspects of the topic
2. Generate 2-3 web queries for surveys, datasets, or benchmarks
3. Avoid queries too similar to ones already run
4. Focus on filling identified gaps
5. Include queries for both seminal works and recent advances
6. Use specific technical terms when appropriate

Output ONLY valid JSON."""

    messages = create_agent_message("search_planner", prompt)
    response = llm.invoke(messages)
    
    queries_data = parse_llm_json(response.content, fallback=None, agent="search_planner")

    if queries_data and isinstance(queries_data, dict):
        pending = []

        for q in queries_data.get("arxiv_queries", []):
            query_text = q.get("query", "")
            if query_text and query_text not in existing_queries:
                pending.append(f"arxiv:{query_text}")

        for q in queries_data.get("web_queries", []):
            query_text = q.get("query", "")
            if query_text and query_text not in existing_queries:
                pending.append(f"web:{query_text}")

        return {
            "pending_queries": pending,
            "phase": "retrieval",
        }

    basic_queries = generate_basic_queries(state.topic, state.research_questions)
    return {
        "pending_queries": basic_queries,
        "phase": "retrieval",
    }


def generate_basic_queries(topic: str, research_questions) -> List[str]:
    """Generate basic queries from topic and research questions."""
    queries = [
        f"arxiv:{topic} survey",
        f"arxiv:{topic} methods",
        f"arxiv:{topic} benchmark dataset",
        f"web:{topic} survey paper",
        f"web:{topic} datasets benchmarks",
    ]

    # Accept either ResearchQuestion objects or plain strings
    for rq in (research_questions or [])[:3]:
        text = rq.text if hasattr(rq, "text") else str(rq)
        key_terms = " ".join([w for w in text.split() if len(w) > 4])[:50]
        if key_terms:
            queries.append(f"arxiv:{key_terms}")

    return queries


