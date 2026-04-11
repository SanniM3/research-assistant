"""Retriever agent - executes searches and retrieves papers."""
from typing import Dict, Any, List
import uuid

from ..models.state import ResearchState, QueryRecord
from ..models.paper import Paper
from ..tools.arxiv import arxiv_search, parse_arxiv_results_to_papers
from ..tools.web_search import web_search


def retriever_node(state: ResearchState) -> Dict[str, Any]:
    """
    Retriever node - executes search queries and retrieves candidate papers.
    
    Responsibilities:
    - Execute pending search queries
    - Parse and normalize results
    - Deduplicate against existing papers
    - Track query execution in audit log
    """
    state.log_action("retriever", "executing_queries", {"count": len(state.pending_queries)})
    
    new_candidates = []
    queries_completed = []
    
    for query_str in state.pending_queries[:10]:  # Limit per iteration
        # Parse query type and text
        if ":" in query_str:
            source, query_text = query_str.split(":", 1)
        else:
            source = "arxiv"
            query_text = query_str
        
        try:
            if source == "arxiv":
                results = execute_arxiv_search(query_text)
                papers = parse_arxiv_results_to_papers(results)
            elif source == "web":
                results = execute_web_search(query_text)
                papers = parse_web_results_to_papers(results, query_text)
            else:
                continue
            
            # Deduplicate against existing AND already-added papers in this batch
            all_existing = state.candidate_papers + new_candidates
            for paper in papers:
                if not is_duplicate(paper, all_existing, state.papers_ingested):
                    new_candidates.append(paper)
                    all_existing.append(paper)
            
            # Record query
            query_record = QueryRecord(
                query_id=str(uuid.uuid4())[:8],
                query_text=query_text,
                source=source,
                results_count=len(results),
                selected_count=len([p for p in papers if not is_duplicate(p, state.candidate_papers, state.papers_ingested)]),
            )
            queries_completed.append(query_record)
            
        except Exception as e:
            state.log_action("retriever", "query_error", {"query": query_str, "error": str(e)})
    
    # Update state
    updated_candidates = state.candidate_papers + new_candidates
    updated_queries = state.queries_run + queries_completed
    
    # Clear pending queries that were processed
    remaining_pending = state.pending_queries[10:]
    
    return {
        "candidate_papers": updated_candidates,
        "queries_run": updated_queries,
        "pending_queries": remaining_pending,
        "phase": "triage",
    }


def execute_arxiv_search(query: str, max_results: int = 10) -> List[Dict]:
    """Execute arXiv search."""
    try:
        results = arxiv_search.invoke({
            "query": query,
            "max_results": max_results,
            "sort_by": "relevance",
        })
        return results
    except Exception as e:
        print(f"arXiv search error: {e}")
        return []


def execute_web_search(query: str, max_results: int = 5) -> List[Dict]:
    """Execute web search."""
    try:
        results = web_search.invoke({
            "query": query,
            "max_results": max_results,
        })
        return results
    except Exception as e:
        print(f"Web search error: {e}")
        return []


def parse_web_results_to_papers(results: List[Dict], query: str) -> List[Paper]:
    """Convert web search results to Paper objects."""
    papers = []
    
    for result in results:
        url = result.get("url", "")
        title = result.get("title", "")
        
        if not title or not url:
            continue
        
        # Try to detect if this is an academic paper
        # (This is a heuristic - web results are less structured)
        is_academic = any(domain in url.lower() for domain in [
            "arxiv", "acl", "ieee", "acm", "springer", "semanticscholar",
            "openreview", "neurips", "mlr.press"
        ])
        
        if is_academic:
            paper = Paper(
                paper_id=Paper.generate_paper_id(title=title),
                title=title,
                url_list=[url],
                abstract=result.get("content", "")[:500],
            )
            papers.append(paper)
    
    return papers


def is_duplicate(paper: Paper, candidates: List[Paper], ingested: Dict[str, Paper]) -> bool:
    """Check if paper is a duplicate of existing papers."""
    # Check by arXiv ID
    if paper.arxiv_id:
        for existing in candidates:
            if existing.arxiv_id and normalize_arxiv_id(existing.arxiv_id) == normalize_arxiv_id(paper.arxiv_id):
                return True
        for existing in ingested.values():
            if existing.arxiv_id and normalize_arxiv_id(existing.arxiv_id) == normalize_arxiv_id(paper.arxiv_id):
                return True
    
    # Check by DOI
    if paper.doi:
        for existing in candidates:
            if existing.doi and existing.doi.lower() == paper.doi.lower():
                return True
        for existing in ingested.values():
            if existing.doi and existing.doi.lower() == paper.doi.lower():
                return True
    
    # Check by title similarity (simple exact match)
    if paper.title:
        normalized_title = paper.title.lower().strip()
        for existing in candidates:
            if existing.title and existing.title.lower().strip() == normalized_title:
                return True
        for existing in ingested.values():
            if existing.title and existing.title.lower().strip() == normalized_title:
                return True
    
    return False


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Normalize arXiv ID by removing version suffix."""
    if arxiv_id:
        return arxiv_id.split("v")[0].strip()
    return ""
