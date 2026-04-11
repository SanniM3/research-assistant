"""Ingestion agent - fetches and processes full paper content."""
from typing import Dict, Any, List

from ..models.state import ResearchState
from ..models.paper import Paper
from ..models.chunk import Chunk, SourceType
from ..tools.document_loaders import load_pdf, load_arxiv_html, load_html, DocumentLoadError
from ..tools.chunking import chunk_document, chunk_text, chunk_with_sections


def ingestion_node(state: ResearchState) -> Dict[str, Any]:
    """
    Ingestion node - fetches full text and creates chunks.
    
    Responsibilities:
    - Fetch HTML + PDF for selected papers
    - Parse and extract text
    - Chunk content for retrieval
    - Store with provenance metadata
    """
    state.log_action("ingestion", "starting", {"selected_count": len(state.selected_papers)})
    
    # Get papers that need ingestion
    papers_to_ingest = [
        p for p in state.candidate_papers 
        if p.paper_id in state.selected_papers and p.paper_id not in state.papers_ingested
    ]
    
    if not papers_to_ingest:
        return {"phase": "extraction"}
    
    new_chunks = dict(state.chunks)
    new_ingested = dict(state.papers_ingested)
    
    for paper in papers_to_ingest[:10]:  # Limit per iteration
        try:
            chunks = ingest_paper(paper)
            
            if chunks:
                # Store chunks
                for chunk in chunks:
                    new_chunks[chunk.chunk_id] = chunk
                
                # Mark paper as ingested
                paper.is_ingested = True
                paper.ingestion_status = "complete"
                new_ingested[paper.paper_id] = paper
                
                state.log_action("ingestion", "paper_ingested", {
                    "paper_id": paper.paper_id,
                    "chunk_count": len(chunks),
                })
            else:
                paper.ingestion_status = "failed"
                state.log_action("ingestion", "paper_failed", {
                    "paper_id": paper.paper_id,
                    "reason": "no_chunks_extracted",
                })
                
        except Exception as e:
            paper.ingestion_status = "error"
            state.log_action("ingestion", "paper_error", {
                "paper_id": paper.paper_id,
                "error": str(e),
            })
    
    return {
        "chunks": new_chunks,
        "papers_ingested": new_ingested,
        "phase": "extraction",
    }


def ingest_paper(paper: Paper) -> List[Chunk]:
    """
    Ingest a single paper, trying multiple sources.
    
    Priority:
    1. arXiv HTML (best structure)
    2. PDF (most complete)
    3. Web HTML (fallback)
    """
    chunks = []
    
    # Try arXiv HTML first
    if paper.arxiv_id:
        chunks = try_arxiv_html_ingestion(paper)
        if chunks:
            return chunks

    # Try PDF
    pdf_url = paper.get_pdf_url()
    if pdf_url:
        chunks = try_pdf_ingestion(paper, pdf_url)
        if chunks:
            return chunks

    # Try web HTML as fallback
    for url in paper.url_list:
        if url and not url.endswith(".pdf"):
            chunks = try_html_ingestion(paper, url)
            if chunks:
                return chunks
    
    # If all else fails, use abstract
    if paper.abstract:
        chunk = Chunk.create(
            paper_id=paper.paper_id,
            text=f"Title: {paper.title}\n\nAbstract: {paper.abstract}",
            source_type=SourceType.WEB_HTML,
            section_path="Abstract",
        )
        chunks = [chunk]
    
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
