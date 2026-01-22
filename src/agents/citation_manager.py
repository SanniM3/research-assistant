"""Citation Manager agent - handles bibliography management with IEEE-style citations."""
import re
from typing import Dict, Any, List, Optional, Tuple
from collections import OrderedDict

from ..models.state import ResearchState
from ..models.paper import Paper
from .base import get_llm


def citation_manager_node(state: ResearchState) -> Dict[str, Any]:
    """
    Citation Manager node - manages bibliography and citations.
    
    Responsibilities:
    - Assign citation numbers to papers
    - Convert internal citations to IEEE-style [N] format
    - Generate numbered reference list
    - Validate citation completeness
    """
    state.log_action("citation_manager", "starting", {
        "papers": len(state.papers_ingested)
    })
    
    # First pass: collect all cited paper IDs from draft sections
    all_content = "\n".join(state.draft_sections.values())
    cited_paper_ids = extract_cited_paper_ids(all_content, state.papers_ingested)
    
    # Assign citation numbers (in order of first appearance)
    paper_to_number = assign_citation_numbers(cited_paper_ids, state.draft_sections)
    
    # Update draft sections with numbered citations
    updated_drafts = {}
    for section_id, content in state.draft_sections.items():
        updated_content = convert_to_ieee_citations(content, paper_to_number, state.papers_ingested)
        updated_drafts[section_id] = updated_content
    
    # Generate numbered reference list
    references = generate_numbered_references(paper_to_number, state.papers_ingested)
    
    # Store as bib_entries (repurposing the field for formatted references)
    bib_entries = {"_references_text": references, "_paper_to_number": paper_to_number}
    
    state.log_action("citation_manager", "completed", {
        "citations_processed": len(paper_to_number)
    })
    
    return {
        "bib_entries": bib_entries,
        "draft_sections": updated_drafts,
    }


def extract_cited_paper_ids(content: str, papers: Dict[str, Paper]) -> List[str]:
    """Extract all paper IDs cited in the content."""
    # Pattern: [@paper_id:chunk_id] or [@paper_id] or existing attempts like [paper_id]
    patterns = [
        r'\[@([^\]:]+)(?::[^\]]+)?\]',  # [@paper_id:chunk_id] or [@paper_id]
        r'\\cite\{([^}]+)\}',            # \cite{citekey}
    ]
    
    cited_ids = []
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            # Handle multiple citations in one reference
            ids = [m.strip() for m in match.split(",")]
            for paper_id in ids:
                if paper_id not in cited_ids:
                    # Try to resolve to actual paper ID
                    resolved = resolve_paper_id(paper_id, papers)
                    if resolved and resolved not in cited_ids:
                        cited_ids.append(resolved)
    
    return cited_ids


def resolve_paper_id(ref: str, papers: Dict[str, Paper]) -> Optional[str]:
    """Resolve a reference to an actual paper ID."""
    # Direct match
    if ref in papers:
        return ref
    
    # Try matching by partial ID
    for paper_id in papers:
        if ref in paper_id or paper_id in ref:
            return paper_id
        # Try matching arxiv ID
        paper = papers[paper_id]
        if paper.arxiv_id and (ref in paper.arxiv_id or paper.arxiv_id in ref):
            return paper_id
    
    # Try matching by citekey-like format (author year)
    ref_lower = ref.lower()
    for paper_id, paper in papers.items():
        if paper.authors and paper.year:
            first_author_last = paper.authors[0].split()[-1].lower()
            if first_author_last in ref_lower and str(paper.year) in ref:
                return paper_id
    
    return None


def assign_citation_numbers(cited_ids: List[str], draft_sections: Dict[str, str]) -> Dict[str, int]:
    """Assign citation numbers in order of first appearance."""
    paper_to_number = OrderedDict()
    citation_counter = 1
    
    # Process sections in order to assign numbers by first appearance
    for section_id in sorted(draft_sections.keys()):
        content = draft_sections[section_id]
        
        # Find all citations in this section
        patterns = [
            r'\[@([^\]:]+)(?::[^\]]+)?\]',
            r'\\cite\{([^}]+)\}',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                ref = match.group(1)
                ids = [m.strip() for m in ref.split(",")]
                for paper_id in ids:
                    if paper_id in cited_ids and paper_id not in paper_to_number:
                        paper_to_number[paper_id] = citation_counter
                        citation_counter += 1
    
    # Add any remaining cited papers that weren't found in sections
    for paper_id in cited_ids:
        if paper_id not in paper_to_number:
            paper_to_number[paper_id] = citation_counter
            citation_counter += 1
    
    return dict(paper_to_number)


def convert_to_ieee_citations(content: str, paper_to_number: Dict[str, int], 
                              papers: Dict[str, Paper]) -> str:
    """Convert internal citations to IEEE-style [N] format."""
    
    def replace_citation(match):
        full_match = match.group(0)
        ref_content = match.group(1)
        
        # Handle multiple citations
        refs = [r.strip() for r in ref_content.split(",")]
        numbers = []
        
        for ref in refs:
            # Try direct match
            if ref in paper_to_number:
                numbers.append(paper_to_number[ref])
            else:
                # Try to resolve
                resolved = resolve_paper_id(ref, papers)
                if resolved and resolved in paper_to_number:
                    numbers.append(paper_to_number[resolved])
        
        if numbers:
            # Sort and format: [1], [2, 3], [1, 4, 7]
            numbers = sorted(set(numbers))
            # Collapse consecutive ranges: [1, 2, 3, 5] -> [1-3, 5]
            formatted = format_citation_numbers(numbers)
            return f"[{formatted}]"
        else:
            # Keep original if can't resolve
            return full_match
    
    # Replace [@paper_id:chunk_id] patterns
    content = re.sub(r'\[@([^\]]+)\]', replace_citation, content)
    
    # Replace \cite{} patterns
    content = re.sub(r'\\cite\{([^}]+)\}', replace_citation, content)
    
    return content


def format_citation_numbers(numbers: List[int]) -> str:
    """Format citation numbers, collapsing consecutive ranges."""
    if not numbers:
        return ""
    
    numbers = sorted(numbers)
    
    if len(numbers) == 1:
        return str(numbers[0])
    
    # Check if we should collapse ranges (for 3+ consecutive numbers)
    ranges = []
    start = numbers[0]
    end = numbers[0]
    
    for n in numbers[1:]:
        if n == end + 1:
            end = n
        else:
            if end - start >= 2:
                ranges.append(f"{start}-{end}")
            elif end - start == 1:
                ranges.append(str(start))
                ranges.append(str(end))
            else:
                ranges.append(str(start))
            start = n
            end = n
    
    # Add the last range
    if end - start >= 2:
        ranges.append(f"{start}-{end}")
    elif end - start == 1:
        ranges.append(str(start))
        ranges.append(str(end))
    else:
        ranges.append(str(start))
    
    return ", ".join(ranges)


def generate_numbered_references(paper_to_number: Dict[str, int], 
                                  papers: Dict[str, Paper]) -> str:
    """Generate IEEE-style numbered reference list."""
    if not paper_to_number:
        return ""
    
    # Sort by citation number
    sorted_papers = sorted(paper_to_number.items(), key=lambda x: x[1])
    
    references = []
    for paper_id, number in sorted_papers:
        paper = papers.get(paper_id)
        if paper:
            ref_text = format_ieee_reference(paper, number)
            references.append(ref_text)
    
    return "\n\n".join(references)


def format_ieee_reference(paper: Paper, number: int) -> str:
    """Format a single reference in IEEE style."""
    parts = []
    
    # [N]
    parts.append(f"[{number}]")
    
    # Authors
    if paper.authors:
        if len(paper.authors) <= 3:
            author_str = ", ".join(paper.authors)
        else:
            author_str = f"{paper.authors[0]} et al."
        parts.append(author_str + ",")
    
    # Title in quotes
    if paper.title:
        parts.append(f'"{paper.title},"')
    
    # Venue/Journal
    if paper.arxiv_id:
        parts.append(f"*arXiv preprint arXiv:{paper.arxiv_id}*,")
    elif paper.venue:
        parts.append(f"*{paper.venue}*,")
    
    # Year
    if paper.year:
        parts.append(f"{paper.year}.")
    
    # DOI or URL
    if paper.doi:
        parts.append(f"DOI: {paper.doi}")
    elif paper.arxiv_id:
        parts.append(f"Available: https://arxiv.org/abs/{paper.arxiv_id}")
    
    return " ".join(parts)


def generate_citekey(paper: Paper) -> str:
    """Generate a citation key for a paper (for BibTeX compatibility)."""
    first_author = ""
    if paper.authors:
        first_author_full = paper.authors[0]
        parts = first_author_full.split()
        first_author = parts[-1] if parts else "Unknown"
        first_author = re.sub(r'[^a-zA-Z]', '', first_author)
    else:
        first_author = "Unknown"
    
    year = paper.year or "XXXX"
    
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
    if paper.arxiv_id:
        entry_type = "article"
        venue = "arXiv preprint arXiv:" + paper.arxiv_id
    elif paper.venue:
        if any(conf in paper.venue.lower() for conf in ["conference", "proceedings", "workshop", "acl", "emnlp", "neurips", "icml"]):
            entry_type = "inproceedings"
            venue = paper.venue
        else:
            entry_type = "article"
            venue = paper.venue
    else:
        entry_type = "misc"
        venue = ""
    
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
