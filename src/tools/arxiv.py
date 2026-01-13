"""arXiv search and paper retrieval tools."""
from typing import List, Optional, Dict, Any
from datetime import datetime
import arxiv
from langchain_core.tools import tool

from ..models.paper import Paper, PaperMetadata, MetadataConfidence


@tool
def arxiv_search(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    categories: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search arXiv for academic papers matching the query.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 10)
        sort_by: Sort order - "relevance", "submitted_date", or "last_updated"
        categories: Optional list of arXiv categories to filter (e.g., ["cs.CL", "cs.AI"])
        date_from: Optional start date filter (YYYY-MM-DD)
        date_to: Optional end date filter (YYYY-MM-DD)
    
    Returns:
        List of paper metadata dictionaries
    """
    # Build query with category filters
    full_query = query
    if categories:
        cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
        full_query = f"({query}) AND ({cat_query})"
    
    # Map sort option
    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submitted_date": arxiv.SortCriterion.SubmittedDate,
        "last_updated": arxiv.SortCriterion.LastUpdatedDate,
    }
    sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)
    
    # Execute search
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=sort_criterion
    )
    
    results = []
    for result in search.results():
        # Apply date filters if specified
        pub_date = result.published
        if date_from:
            from_dt = datetime.strptime(date_from, "%Y-%m-%d")
            if pub_date.replace(tzinfo=None) < from_dt:
                continue
        if date_to:
            to_dt = datetime.strptime(date_to, "%Y-%m-%d")
            if pub_date.replace(tzinfo=None) > to_dt:
                continue
        
        # Extract arXiv ID from entry_id
        arxiv_id = result.entry_id.split("/")[-1]
        
        paper_dict = {
            "arxiv_id": arxiv_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "published": result.published.isoformat(),
            "updated": result.updated.isoformat() if result.updated else None,
            "categories": result.categories,
            "primary_category": result.primary_category,
            "pdf_url": result.pdf_url,
            "html_url": f"https://arxiv.org/abs/{arxiv_id}",
            "doi": result.doi,
            "comment": result.comment,
        }
        results.append(paper_dict)
    
    return results


def fetch_arxiv_paper(arxiv_id: str) -> Optional[Paper]:
    """
    Fetch full paper metadata from arXiv.
    
    Args:
        arxiv_id: arXiv identifier (e.g., "2301.12345")
    
    Returns:
        Paper object with full metadata, or None if not found
    """
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(search.results())
        
        if not results:
            return None
        
        result = results[0]
        
        # Extract year from published date
        year = result.published.year if result.published else None
        
        # Create Paper object
        paper = Paper(
            paper_id=Paper.generate_paper_id(arxiv_id=arxiv_id),
            title=result.title,
            authors=[author.name for author in result.authors],
            year=year,
            venue="arXiv",
            doi=result.doi,
            arxiv_id=arxiv_id,
            url_list=[
                result.pdf_url,
                f"https://arxiv.org/abs/{arxiv_id}",
            ],
            abstract=result.summary,
            metadata_confidence=MetadataConfidence.HIGH,
            metadata=PaperMetadata(
                categories=result.categories,
            )
        )
        
        return paper
        
    except Exception as e:
        print(f"Error fetching arXiv paper {arxiv_id}: {e}")
        return None


def parse_arxiv_results_to_papers(results: List[Dict[str, Any]]) -> List[Paper]:
    """
    Convert arXiv search results to Paper objects.
    
    Args:
        results: List of result dictionaries from arxiv_search
    
    Returns:
        List of Paper objects
    """
    papers = []
    for result in results:
        arxiv_id = result.get("arxiv_id", "")
        
        # Parse year from published date
        year = None
        if result.get("published"):
            try:
                pub_dt = datetime.fromisoformat(result["published"].replace("Z", "+00:00"))
                year = pub_dt.year
            except (ValueError, AttributeError):
                pass
        
        paper = Paper(
            paper_id=Paper.generate_paper_id(arxiv_id=arxiv_id),
            title=result.get("title", ""),
            authors=result.get("authors", []),
            year=year,
            venue="arXiv",
            doi=result.get("doi"),
            arxiv_id=arxiv_id,
            url_list=[
                result.get("pdf_url", ""),
                result.get("html_url", ""),
            ],
            abstract=result.get("abstract", ""),
            metadata_confidence=MetadataConfidence.HIGH,
            metadata=PaperMetadata(
                categories=result.get("categories", []),
            )
        )
        papers.append(paper)
    
    return papers
