"""Citation Manager agent - handles bibliography management with IEEE-style citations."""
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from collections import OrderedDict

from ..models.state import ResearchState
from ..models.paper import Paper


def _strip_chunk_suffix(ref: str) -> str:
    """Strip an optional chunk_id suffix from a citation reference.

    Paper IDs may contain colons (e.g. ``arxiv:1706.03762``).  The
    convention ``[@paper_id:chunk_id]`` uses a *chunk-style* suffix
    (``chunk_`` prefix).  We only strip the last colon segment if it
    looks like a chunk id; otherwise we keep the full string.
    """
    if ":" not in ref:
        return ref
    head, tail = ref.rsplit(":", 1)
    if tail.startswith("chunk_") or tail.startswith("chk_"):
        return head
    return ref


def citation_manager_node(state: ResearchState) -> Dict[str, Any]:
    """
    Citation Manager node - manages bibliography and citations.
    
    Responsibilities:
    - Assign citation numbers to papers
    - Convert internal citations to IEEE-style [N] format
    - Generate numbered reference list
    - Validate citation completeness
    """
    papers = state.kb().papers_map()
    state.log_action("citation_manager", "starting", {"papers": len(papers)})
    
    # First pass: collect all cited paper IDs from draft sections
    all_content = "\n".join(state.draft_sections.values())
    cited_paper_ids, citation_mapping = extract_cited_paper_ids(all_content, papers)
    
    # Collect unresolved citations (these will be dropped, never faked)
    unresolved = [k for k, v in citation_mapping.items() if v is None]
    
    # If no citations were resolved, include all reviewed papers as references
    if not cited_paper_ids and papers:
        cited_paper_ids = list(papers.keys())
    
    # Assign citation numbers (in order of first appearance)
    paper_to_number, citation_to_number = assign_citation_numbers(
        cited_paper_ids, citation_mapping, state.draft_sections, papers
    )
    
    # Update draft sections with numbered citations (unresolved refs removed)
    updated_drafts = {}
    for section_id, content in state.draft_sections.items():
        updated_content = convert_to_ieee_citations(
            content, paper_to_number, citation_to_number, papers
        )
        updated_drafts[section_id] = updated_content
    
    # Generate numbered reference list (resolved papers only)
    references = generate_numbered_references(paper_to_number, papers)
    
    if unresolved:
        state.log_action("citation_manager", "unresolved_citations_dropped", {
            "count": len(unresolved),
            "sample": unresolved[:10],
        })
    
    # Store as bib_entries — serialize paper_to_number as a JSON string to
    # avoid Pydantic / LangGraph checkpoint validation issues with nested dicts.
    bib_entries = {
        "_references_text": references,
        "_paper_to_number": json.dumps(paper_to_number),
    }
    
    state.log_action("citation_manager", "completed", {
        "citations_resolved": len(paper_to_number),
        "citations_unresolved": len(unresolved),
    })
    
    return {
        "bib_entries": bib_entries,
        "draft_sections": updated_drafts,
    }


def extract_cited_paper_ids(content: str, papers: Dict[str, Paper]) -> Tuple[List[str], Dict[str, str]]:
    """
    Extract all paper IDs cited in the content.
    
    Returns:
        Tuple of (resolved_paper_ids, citation_to_paper_mapping)
    """
    # [@paper_id] or [@paper_id:chunk_id] — paper_id may contain colons (e.g. arxiv:1706.03762)
    # so we capture the full content and split on the LAST colon for chunk_id.
    patterns = [
        r'\[@([^\]]+)\]',        # [@...] — full content between brackets
        r'\\cite\{([^}]+)\}',   # \cite{citekey}
    ]

    cited_ids = []
    citation_mapping = {}

    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            ids = [m.strip() for m in match.split(",")]
            for raw_ref in ids:
                citation_text = _strip_chunk_suffix(raw_ref)
                if citation_text in citation_mapping:
                    continue
                    
                # Try to resolve to actual paper ID
                resolved = resolve_paper_id(citation_text, papers)
                if resolved:
                    citation_mapping[citation_text] = resolved
                    if resolved not in cited_ids:
                        cited_ids.append(resolved)
                else:
                    # Keep unresolved citations for later handling
                    citation_mapping[citation_text] = None
    
    return cited_ids, citation_mapping


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Normalize arxiv ID by removing version suffix and 'arxiv:' prefix."""
    if not arxiv_id:
        return ""
    # Remove 'arxiv:' prefix if present
    normalized = arxiv_id.lower().replace("arxiv:", "").strip()
    # Remove version suffix (e.g., v1, v2)
    if "v" in normalized:
        normalized = normalized.split("v")[0]
    return normalized


def resolve_paper_id(ref: str, papers: Dict[str, Paper]) -> Optional[str]:
    """Resolve a citation reference to a paper ID by EXACT / normalized id only.

    Fuzzy substring and title-keyword matching were removed deliberately: they
    routinely misattributed citations to the wrong paper, which is fatal for a
    system whose value is correct grounding. The synthesizer cites from a closed
    set of paper ids, so exact/normalized-id resolution is sufficient; anything
    that does not resolve is dropped rather than guessed.
    """
    # Direct match
    if ref in papers:
        return ref

    # Normalized arXiv id match (handles arxiv: prefix and version suffixes)
    ref_normalized = normalize_arxiv_id(ref)
    if not ref_normalized:
        return None
    for paper_id, paper in papers.items():
        if normalize_arxiv_id(paper_id) == ref_normalized:
            return paper_id
        if paper.arxiv_id and normalize_arxiv_id(paper.arxiv_id) == ref_normalized:
            return paper_id

    return None


def assign_citation_numbers(cited_ids: List[str], citation_mapping: Dict[str, Optional[str]],
                            draft_sections: Dict[str, str], 
                            papers: Dict[str, Paper]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Assign citation numbers in order of first appearance.
    
    Returns:
        Tuple of (paper_id -> number, original_citation_text -> number)
    """
    paper_to_number = OrderedDict()
    citation_to_number = {}  # Maps original citation text to number
    citation_counter = 1
    
    # Process sections in order to assign numbers by first appearance
    for section_id in sorted(draft_sections.keys()):
        content = draft_sections[section_id]
        
        patterns = [
            r'\[@([^\]]+)\]',
            r'\\cite\{([^}]+)\}',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content):
                ref = match.group(1)
                ids = [_strip_chunk_suffix(m.strip()) for m in ref.split(",")]
                for citation_text in ids:
                    # Try to resolve this citation
                    resolved = citation_mapping.get(citation_text)
                    if not resolved:
                        resolved = resolve_paper_id(citation_text, papers)
                    
                    if resolved and resolved not in paper_to_number:
                        paper_to_number[resolved] = citation_counter
                        citation_counter += 1
                    
                    # Map the original citation text to the number
                    if resolved and resolved in paper_to_number:
                        citation_to_number[citation_text] = paper_to_number[resolved]
    
    # Add any remaining cited papers that weren't found in sections
    for paper_id in cited_ids:
        if paper_id not in paper_to_number:
            paper_to_number[paper_id] = citation_counter
            citation_counter += 1
    
    return dict(paper_to_number), citation_to_number


def convert_to_ieee_citations(content: str, paper_to_number: Dict[str, int],
                              citation_to_number: Dict[str, int],
                              papers: Dict[str, Paper]) -> str:
    """Convert internal citations to IEEE-style [N] format."""
    
    def replace_citation(match):
        full_match = match.group(0)
        ref_content = match.group(1)
        
        refs = [_strip_chunk_suffix(r.strip()) for r in ref_content.split(",")]
        numbers = []

        for ref in refs:
            # First try the pre-computed mapping
            if ref in citation_to_number:
                numbers.append(citation_to_number[ref])
            # Try direct match in paper_to_number
            elif ref in paper_to_number:
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
            # Can't resolve - drop the citation marker cleanly rather than
            # leaving debugging noise or fabricating a reference.
            return ""
    
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
        # If no citations were resolved, generate references from all papers
        # This is a fallback to ensure we don't have an empty reference section
        if papers:
            references = []
            for i, (paper_id, paper) in enumerate(papers.items(), 1):
                ref_text = format_ieee_reference(paper, i)
                references.append(ref_text)
            return "\n\n".join(references[:20])  # Limit to 20 references
        return "*No references available.*"
    
    sorted_papers = sorted(paper_to_number.items(), key=lambda x: x[1])

    references = []
    for paper_id, number in sorted_papers:
        paper = papers.get(paper_id)
        if paper:
            ref_text = format_ieee_reference(paper, number)
            if ref_text:
                references.append(ref_text)

    return "\n\n".join(references) if references else "*No references available.*"


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
    
    return " ".join(parts)


def format_placeholder_reference(citation_text: str, number: int) -> str:
    """Format a placeholder reference for unresolved citations."""
    # Try to extract useful info from the citation text
    # Common patterns: arxiv:1706.03762, doi:10.xxxx, author2020keyword
    
    parts = [f"[{number}]"]
    
    if "arxiv:" in citation_text.lower():
        # Extract arxiv ID
        arxiv_id = citation_text.lower().replace("arxiv:", "").strip()
        parts.append(f"*arXiv preprint arXiv:{arxiv_id}*")
        parts.append(f"(https://arxiv.org/abs/{arxiv_id})")
    elif "doi:" in citation_text.lower():
        doi = citation_text.replace("doi:", "").strip()
        parts.append(f"DOI: {doi}")
    else:
        # Try to parse author-year format like "vaswani2017attention"
        import re
        match = re.match(r'([a-z]+)(\d{4})([a-z]*)', citation_text.lower())
        if match:
            author = match.group(1).capitalize()
            year = match.group(2)
            keyword = match.group(3) if match.group(3) else ""
            parts.append(f"{author} et al., {year}.")
            if keyword:
                parts.append(f"(Reference: {citation_text})")
        else:
            parts.append(f"Reference: {citation_text}")
    
    return " ".join(parts)


