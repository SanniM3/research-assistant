"""Ingestion agent - fetches and processes full paper content into the KB."""
import time
from typing import Dict, Any, List, Tuple

from ..models.state import ResearchState
from ..models.paper import Paper
from ..models.chunk import Chunk, SourceType
from ..tools.document_loaders import load_pdf, load_arxiv_html, load_html, DocumentLoadError
from ..tools.chunking import chunk_document, chunk_text, chunk_with_sections, detect_language
from ..utils.logging import get_logger

_logger = get_logger("ingestion")


def ingestion_node(state: ResearchState) -> Dict[str, Any]:
    """
    Ingestion node - fetches full text, creates chunks, and writes them to the
    persistent knowledge base (which also embeds new chunks for RAG).

    Papers that only yield an abstract are recorded with
    ``ingestion_status="abstract_only"`` so they are NOT counted as fully
    reviewed and do not inflate coverage statistics.
    """
    kb = state.kb()
    state.log_action("ingestion", "starting", {"selected_count": len(state.selected_papers)})

    selected = set(state.selected_papers)
    papers_to_ingest = [
        p for p in state.candidate_papers
        if p.paper_id in selected and kb.get_paper(p.paper_id) is None
    ]

    if not papers_to_ingest:
        return {"phase": "extraction"}

    batch = papers_to_ingest[:10]  # bound work per iteration
    total = len(batch)
    _logger.info("Ingesting %d paper(s) (downloading full text + embedding chunks)...", total)

    for idx, paper in enumerate(batch, 1):
        t0 = time.time()
        _logger.info("  [%d/%d] fetching %s - %.60s", idx, total, paper.paper_id, paper.title or "")
        try:
            chunks, status = ingest_paper(paper)

            if chunks:
                kb.upsert_chunks(chunks)
                _logger.info("  [%d/%d] done %s: %d chunks (%s) in %.1fs",
                             idx, total, paper.paper_id, len(chunks), status, time.time() - t0)
                paper.is_ingested = True
                paper.ingestion_status = status
                paper.language = chunks[0].metadata.language or paper.language
                kb.upsert_paper(paper)

                state.log_action("ingestion", "paper_ingested", {
                    "paper_id": paper.paper_id,
                    "chunk_count": len(chunks),
                    "status": status,
                })
            else:
                paper.ingestion_status = "failed"
                kb.upsert_paper(paper)
                _logger.warning("  [%d/%d] no full text for %s", idx, total, paper.paper_id)
                state.log_action("ingestion", "paper_failed", {
                    "paper_id": paper.paper_id,
                    "reason": "no_chunks_extracted",
                })

        except Exception as e:
            paper.ingestion_status = "error"
            _logger.warning("  [%d/%d] error ingesting %s: %s", idx, total, paper.paper_id, e)
            state.log_action("ingestion", "paper_error", {
                "paper_id": paper.paper_id,
                "error": str(e),
            })

    return {"phase": "extraction"}


def ingest_paper(paper: Paper) -> Tuple[List[Chunk], str]:
    """
    Ingest a single paper, trying multiple full-text sources before falling back
    to the abstract.

    Returns (chunks, status) where status is "complete" for real full text and
    "abstract_only" when only the abstract was available.
    """
    # Try arXiv HTML first (best structure)
    if paper.arxiv_id:
        chunks = try_arxiv_html_ingestion(paper)
        if chunks:
            return _finalise(chunks), "complete"

    # Try PDF
    pdf_url = paper.get_pdf_url()
    if pdf_url:
        chunks = try_pdf_ingestion(paper, pdf_url)
        if chunks:
            return _finalise(chunks), "complete"

    # Try web HTML as fallback
    for url in paper.url_list:
        if url and not url.endswith(".pdf"):
            chunks = try_html_ingestion(paper, url)
            if chunks:
                return _finalise(chunks), "complete"

    # Last resort: abstract only
    if paper.abstract:
        chunk = Chunk.create(
            paper_id=paper.paper_id,
            text=f"Title: {paper.title}\n\nAbstract: {paper.abstract}",
            source_type=SourceType.WEB_HTML,
            section_path="Abstract",
        )
        return _finalise([chunk]), "abstract_only"

    return [], "failed"


def _finalise(chunks: List[Chunk]) -> List[Chunk]:
    """Tag each chunk with a detected language for cross-lingual retrieval."""
    for chunk in chunks:
        try:
            chunk.metadata.language = detect_language(chunk.text)
        except Exception:
            pass
    return chunks


def try_arxiv_html_ingestion(paper: Paper) -> List[Chunk]:
    """Try to ingest from arXiv HTML."""
    if not paper.arxiv_id:
        return []
    try:
        text, metadata = load_arxiv_html(paper.arxiv_id)
        if text and len(text) > 100:
            return chunk_with_sections(
                text=text,
                paper_id=paper.paper_id,
                source_type=SourceType.ARXIV_HTML,
            )
    except DocumentLoadError:
        pass
    except Exception as e:
        print(f"arXiv HTML ingestion error for {paper.arxiv_id}: {e}")
    return []


def try_pdf_ingestion(paper: Paper, pdf_url: str) -> List[Chunk]:
    """Try to ingest from PDF."""
    try:
        documents, metadata = load_pdf(pdf_url)
        if documents:
            return chunk_document(
                documents=documents,
                paper_id=paper.paper_id,
                source_type=SourceType.PDF_TEXT,
            )
    except DocumentLoadError:
        pass
    except Exception as e:
        print(f"PDF ingestion error for {paper.paper_id}: {e}")
    return []


def try_html_ingestion(paper: Paper, url: str) -> List[Chunk]:
    """Try to ingest from web HTML."""
    try:
        text, metadata = load_html(url)
        if text and len(text) > 100:
            return chunk_text(
                text=text,
                paper_id=paper.paper_id,
                source_type=SourceType.WEB_HTML,
            )
    except DocumentLoadError:
        pass
    except Exception as e:
        print(f"HTML ingestion error for {paper.paper_id}: {e}")
    return []
