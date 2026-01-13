"""Search planner agent - generates search queries."""
import json
from typing import Dict, Any, List
from langchain_core.messages import AIMessage

from ..models.state import ResearchState
from ..models.issue import IssueCategory
from .base import get_llm, create_agent_message


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
    ]]
    
    # Build context about what we already have
    ingested_titles = [p.title for p in state.papers_ingested.values()][:20]
    
    prompt = f"""Generate search queries for academic research on:

TOPIC: {state.topic}
SCOPE: {state.scope}

RESEARCH QUESTIONS:
{json.dumps(state.research_questions, indent=2)}

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
    
    try:
        queries_data = json.loads(response.content)
        
        # Extract queries as simple list for pending_queries
        pending = []
        
        for q in queries_data.get("arxiv_queries", []):
            query_text = q.get("query", "")
            if query_text and query_text not in existing_queries:
                # Format: "arxiv:query_text"
                pending.append(f"arxiv:{query_text}")
        
        for q in queries_data.get("web_queries", []):
            query_text = q.get("query", "")
            if query_text and query_text not in existing_queries:
                # Format: "web:query_text"
                pending.append(f"web:{query_text}")
        
        return {
            "pending_queries": pending,
            "phase": "retrieval",
        }
        
    except json.JSONDecodeError:
        # Fallback: generate basic queries from topic
        basic_queries = generate_basic_queries(state.topic, state.research_questions)
        return {
            "pending_queries": basic_queries,
            "phase": "retrieval",
        }


def generate_basic_queries(topic: str, research_questions: List[str]) -> List[str]:
    """Generate basic queries from topic and research questions."""
    queries = [
        f"arxiv:{topic} survey",
        f"arxiv:{topic} methods",
        f"arxiv:{topic} benchmark dataset",
        f"web:{topic} survey paper",
        f"web:{topic} datasets benchmarks",
    ]
    
    # Add queries from research questions
    for rq in research_questions[:3]:
        # Extract key terms
        key_terms = " ".join([w for w in rq.split() if len(w) > 4])[:50]
        if key_terms:
            queries.append(f"arxiv:{key_terms}")
    
    return queries


def generate_gap_filling_queries(issues: List) -> List[str]:
    """Generate queries specifically to address identified gaps."""
    queries = []
    
    for issue in issues:
        # Use suggested queries from issue if available
        if issue.suggested_queries:
            for sq in issue.suggested_queries[:2]:
                queries.append(f"arxiv:{sq}")
        
        # Generate based on category
        if issue.category == IssueCategory.MISSING_SEMINAL:
            queries.append(f"arxiv:{issue.description} seminal foundational")
        elif issue.category == IssueCategory.MISSING_RECENT:
            queries.append(f"arxiv:{issue.description} 2023 2024 recent")
        elif issue.category == IssueCategory.BENCHMARK_GAP:
            queries.append(f"web:{issue.description} benchmark dataset evaluation")
    
    return queries
