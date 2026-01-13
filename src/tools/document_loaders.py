"""Document loading tools for PDFs and HTML."""
from typing import List, Optional, Dict, Any, Tuple
import os
import tempfile
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from ..models.chunk import SourceType


class DocumentLoadError(Exception):
    """Error loading a document."""
    pass


def load_pdf(source: str, timeout: int = 30) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Load and parse a PDF document.
    
    Args:
        source: URL or local file path to PDF
        timeout: Request timeout for URL downloads
    
    Returns:
        Tuple of (list of Document objects, metadata dict)
    """
    temp_file = None
    
    try:
        # Handle URL sources
        if source.startswith(("http://", "https://")):
            response = requests.get(source, timeout=timeout)
            response.raise_for_status()
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            temp_file.write(response.content)
            temp_file.close()
            file_path = temp_file.name
        else:
            file_path = source
        
        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Extract metadata
        metadata = {
            "source": source,
            "source_type": SourceType.PDF_TEXT.value,
            "page_count": len(documents),
            "total_chars": sum(len(doc.page_content) for doc in documents),
        }
        
        return documents, metadata
        
    except requests.RequestException as e:
        raise DocumentLoadError(f"Failed to download PDF: {e}")
    except Exception as e:
        raise DocumentLoadError(f"Failed to load PDF: {e}")
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def load_html(url: str, timeout: int = 30) -> Tuple[str, Dict[str, Any]]:
    """
    Load and parse an HTML page.
    
    Args:
        url: URL to fetch
        timeout: Request timeout
    
    Returns:
        Tuple of (extracted text content, metadata dict)
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
        
        # Extract main content
        # Try common content containers
        main_content = None
        for selector in ["main", "article", '[role="main"]', ".content", "#content"]:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            # Fall back to body
            body = soup.find("body")
            text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)
        
        metadata = {
            "source": url,
            "source_type": SourceType.WEB_HTML.value,
            "title": title,
            "total_chars": len(text),
        }
        
        return text, metadata
        
    except requests.RequestException as e:
        raise DocumentLoadError(f"Failed to fetch HTML: {e}")
    except Exception as e:
        raise DocumentLoadError(f"Failed to parse HTML: {e}")


def load_arxiv_html(arxiv_id: str, timeout: int = 30) -> Tuple[str, Dict[str, Any]]:
    """
    Load and parse arXiv HTML version of a paper.
    
    Args:
        arxiv_id: arXiv identifier (e.g., "2301.12345")
        timeout: Request timeout
    
    Returns:
        Tuple of (extracted text content, metadata dict)
    """
    # arXiv HTML URL format
    html_url = f"https://arxiv.org/html/{arxiv_id}"
    
    try:
        response = requests.get(html_url, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract title
        title = ""
        title_tag = soup.find("h1", class_="ltx_title")
        if title_tag:
            title = title_tag.get_text(strip=True)
        
        # Extract sections with their headings
        sections = []
        current_section = {"heading": "Abstract", "content": []}
        
        # Get abstract
        abstract = soup.find("div", class_="ltx_abstract")
        if abstract:
            current_section["content"].append(abstract.get_text(strip=True))
        sections.append(current_section)
        
        # Get main content sections
        for section in soup.find_all("section", class_="ltx_section"):
            heading_tag = section.find(["h2", "h3", "h4"])
            heading = heading_tag.get_text(strip=True) if heading_tag else "Section"
            
            # Get paragraphs
            paragraphs = []
            for p in section.find_all("p", class_="ltx_p"):
                text = p.get_text(strip=True)
                if text:
                    paragraphs.append(text)
            
            if paragraphs:
                sections.append({
                    "heading": heading,
                    "content": paragraphs
                })
        
        # Combine into text
        text_parts = []
        for section in sections:
            text_parts.append(f"\n## {section['heading']}\n")
            text_parts.extend(section["content"])
        
        text = "\n\n".join(text_parts)
        
        metadata = {
            "source": html_url,
            "source_type": SourceType.ARXIV_HTML.value,
            "arxiv_id": arxiv_id,
            "title": title,
            "total_chars": len(text),
            "section_count": len(sections),
        }
        
        return text, metadata
        
    except requests.RequestException as e:
        # HTML version might not be available, fall back gracefully
        raise DocumentLoadError(f"arXiv HTML not available for {arxiv_id}: {e}")
    except Exception as e:
        raise DocumentLoadError(f"Failed to parse arXiv HTML: {e}")


def extract_sections_from_html(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """
    Extract hierarchical sections from HTML.
    
    Returns list of dicts with heading, level, and content.
    """
    sections = []
    
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        level = int(heading.name[1])
        heading_text = heading.get_text(strip=True)
        
        # Get content until next heading
        content = []
        for sibling in heading.find_next_siblings():
            if sibling.name and sibling.name.startswith("h"):
                break
            if sibling.name in ["p", "div", "ul", "ol", "table"]:
                content.append(sibling.get_text(strip=True))
        
        sections.append({
            "heading": heading_text,
            "level": level,
            "content": "\n".join(content),
        })
    
    return sections
