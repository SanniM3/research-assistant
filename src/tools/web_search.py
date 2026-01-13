"""Web search tools using Tavily."""
from typing import List, Dict, Any, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool


@tool
def web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Search the web using Tavily for academic and general content.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default 5)
        search_depth: "basic" or "advanced" for deeper search
        include_domains: Optional list of domains to include
        exclude_domains: Optional list of domains to exclude
    
    Returns:
        List of search result dictionaries with url, content, title
    """
    tavily = TavilySearchResults(
        max_results=max_results,
        search_depth=search_depth,
        include_domains=include_domains or [],
        exclude_domains=exclude_domains or [],
    )
    
    results = tavily.invoke(query)
    
    # Normalize results
    normalized = []
    for result in results:
        normalized.append({
            "url": result.get("url", ""),
            "title": result.get("title", ""),
            "content": result.get("content", ""),
            "score": result.get("score", 0.0),
        })
    
    return normalized


def search_academic_sources(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search specifically for academic sources.
    
    Includes academic domains like Google Scholar, Semantic Scholar,
    ACL Anthology, etc.
    """
    academic_domains = [
        "scholar.google.com",
        "semanticscholar.org",
        "aclanthology.org",
        "papers.nips.cc",
        "proceedings.mlr.press",
        "openreview.net",
        "ieee.org",
        "acm.org",
    ]
    
    return web_search.invoke({
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_domains": academic_domains,
    })


def search_for_datasets(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search specifically for datasets and benchmarks.
    """
    dataset_domains = [
        "huggingface.co",
        "kaggle.com",
        "paperswithcode.com",
        "github.com",
    ]
    
    enhanced_query = f"{query} dataset benchmark"
    
    return web_search.invoke({
        "query": enhanced_query,
        "max_results": max_results,
        "include_domains": dataset_domains,
    })


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    Format search results as a readable string.
    
    Args:
        results: List of search result dictionaries
    
    Returns:
        Formatted string representation
    """
    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(
            f"{i}. [{result.get('title', 'No title')}]({result.get('url', '')})\n"
            f"   {result.get('content', 'No content')[:200]}..."
        )
    return "\n\n".join(formatted)
