"""Citation Manager agent - handles bibliography management."""
import re
from typing import Dict, Any, List, Optional

from ..models.state import ResearchState
from ..models.paper import Paper
from .base import get_llm


def citation_manager_node(state: ResearchState) -> Dict[str, Any]:
    """
    Citation Manager node - manages bibliography and citations.
    
    Responsibilities:
    - Generate BibTeX entries for all cited papers
    - Normalize citation keys
    - Convert internal citations to final format
    - Validate citation completeness
    """
    state.log_action("citation_manager", "starting", {
        "papers": len(state.papers_ingested)
    })
    
    # Generate BibTeX entries
    bib_entries = {}
    for paper_id, paper in state.papers_ingested.items():
        citekey = generate_citekey(paper)
        bibtex = generate_bibtex(paper, citekey)
        bib_entries[citekey] = bibtex
    
    # Update draft sections with normalized citations
    updated_drafts = {}
    for section_id, content in state.draft_sections.items():
        updated_content = normalize_citations(content, state.papers_ingested, bib_entries)
        updated_drafts[section_id] = updated_content
    
    state.log_action("citation_manager", "completed", {
        "bib_entries": len(bib_entries)
    })
    
    return {
        "bib_entries": bib_entries,
        "draft_sections": updated_drafts,
    }


def generate_citekey(paper: Paper) -> str:
    """Generate a citation key for a paper."""
    # Format: FirstAuthorYear
    first_author = ""
    if paper.authors:
        # Extract last name of first author
        first_author_full = paper.authors[0]
        parts = first_author_full.split()
        first_author = parts[-1] if parts else "Unknown"
        # Clean up
        first_author = re.sub(r'[^a-zA-Z]', '', first_author)
    else:
        first_author = "Unknown"
    
    year = paper.year or "XXXX"
    
    # Add first word of title to disambiguate
    title_word = ""
    if paper.title:
        words = paper.title.split()
        for word in words:
            if len(word) > 3 and word.lower() not in ["the", "and", "for", "with"]:
                title_word = re.sub(r'[^a-zA-Z]', '', word)
                break
    
    return f"{first_author}{year}{title_word}".lower()


def generate_bibtex(paper: Paper, citekey: str) -> str:
    """Generate BibTeX entry for a paper."""
    # Determine entry type
    if paper.arxiv_id:
        entry_type = "article"
        venue = "arXiv preprint arXiv:" + paper.arxiv_id
    elif paper.venue:
        # Heuristic: conferences vs journals
        if any(conf in paper.venue.lower() for conf in ["conference", "proceedings", "workshop", "acl", "emnlp", "neurips", "icml"]):
            entry_type = "inproceedings"
            venue = paper.venue
        else:
            entry_type = "article"
            venue = paper.venue
    else:
        entry_type = "misc"
        venue = ""
    
    # Build BibTeX
    lines = [f"@{entry_type}{{{citekey},"]
    lines.append(f'  title = {{{escape_bibtex(paper.title)}}},')
    
    if paper.authors:
        authors_str = " and ".join(paper.authors)
        lines.append(f'  author = {{{escape_bibtex(authors_str)}}},')
    
    if paper.year:
        lines.append(f'  year = {{{paper.year}}},')
    
    if venue:
        if entry_type == "inproceedings":
            lines.append(f'  booktitle = {{{escape_bibtex(venue)}}},')
        else:
            lines.append(f'  journal = {{{escape_bibtex(venue)}}},')
    
    if paper.doi:
        lines.append(f'  doi = {{{paper.doi}}},')
    
    if paper.arxiv_id:
        lines.append(f'  eprint = {{{paper.arxiv_id}}},')
        lines.append('  archiveprefix = {arXiv},')
    
    url = paper.get_primary_url()
    if url:
        lines.append(f'  url = {{{url}}},')
    
    lines.append("}")
    
    return "\n".join(lines)


def escape_bibtex(text: str) -> str:
    """Escape special characters for BibTeX."""
    if not text:
        return ""
    
    # Replace special characters
    replacements = [
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ]
    
    result = text
    for old, new in replacements:
        result = result.replace(old, new)
    
    return result


def normalize_citations(content: str, papers: Dict[str, Paper], 
                        bib_entries: Dict[str, str]) -> str:
    """
    Normalize internal citations to standard format.
    
    Converts [@paper_id:chunk_id] to \\cite{citekey} or [N] format.
    """
    # Build paper_id to citekey mapping
    paper_to_citekey = {}
    for paper_id, paper in papers.items():
        citekey = generate_citekey(paper)
        paper_to_citekey[paper_id] = citekey
    
    # Pattern: [@paper_id:chunk_id] or [@paper_id]
    pattern = r'\[@([^\]:]+)(?::([^\]]+))?\]'
    
    def replace_citation(match):
        paper_id = match.group(1)
        # chunk_id = match.group(2)  # Not used in final format
        
        # Find citekey
        citekey = paper_to_citekey.get(paper_id)
        
        if not citekey:
            # Try to match by partial ID
            for pid, key in paper_to_citekey.items():
                if paper_id in pid or pid in paper_id:
                    citekey = key
                    break
        
        if citekey:
            return f"\\cite{{{citekey}}}"
        else:
            return f"[{paper_id}]"  # Keep as placeholder
    
    return re.sub(pattern, replace_citation, content)


def compile_bibliography(bib_entries: Dict[str, str]) -> str:
    """Compile all BibTeX entries into a bibliography file."""
    entries = list(bib_entries.values())
    return "\n\n".join(entries)


def validate_citations(content: str, bib_entries: Dict[str, str]) -> List[str]:
    """Validate that all citations in content have BibTeX entries."""
    missing = []
    
    # Find all \cite{} references
    pattern = r'\\cite\{([^}]+)\}'
    cites = re.findall(pattern, content)
    
    for cite in cites:
        # Handle multiple citations in one \cite{}
        keys = [k.strip() for k in cite.split(",")]
        for key in keys:
            if key not in bib_entries:
                missing.append(key)
    
    return list(set(missing))
