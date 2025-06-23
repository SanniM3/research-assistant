from typing import Dict, Any, List
from langchain.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.utilities import PubMedAPIWrapper
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import arxiv
from Bio import Entrez
import os
from dotenv import load_dotenv

load_dotenv()

# class ArxivTool(BaseTool):
#     name = "arxiv_search"
#     description = "Search for academic papers on arXiv"
    
#     def _run(self, query: str) -> List[Dict[str, Any]]:
#         search = arxiv.Search(
#             query=query,
#             max_results=5,
#             sort_by=arxiv.SortCriterion.Relevance
#         )
        
#         results = []
#         for result in search.results():
#             results.append({
#                 "title": result.title,
#                 "authors": [author.name for author in result.authors],
#                 "abstract": result.summary,
#                 "url": result.entry_id,
#                 "published": result.published.isoformat()
#             })
        # return results

def arxiv_search_tool(query: str, max_results: int = 5) -> list:
    """
    Search arXiv for academic papers matching the query.

    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to return.

    Returns:
        list: A list of dictionaries with paper metadata.
    """
    print(f"Arxiv search tool called with query: {query}")
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for result in search.results():
        results.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "url": result.entry_id,
            "published": result.published.isoformat()
        })
    print(f'Arxiv search returned: {results}')
    return results

# class PubMedTool(BaseTool):
#     name = "pubmed_search"
#     description = "Search for medical research papers on PubMed"
    
#     def _run(self, query: str) -> List[Dict[str, Any]]:
#         Entrez.email = os.getenv("ENTREZ_EMAIL", "your-email@example.com")
        
#         # Search PubMed
#         handle = Entrez.esearch(db="pubmed", term=query, retmax=5)
#         record = Entrez.read(handle)
#         handle.close()
        
#         id_list = record["IdList"]
        
#         # Fetch details for each paper
#         handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
#         records = Entrez.read(handle)
#         handle.close()
        
#         results = []
#         for record in records:
#             results.append({
#                 "title": record.get("TI", ""),
#                 "authors": record.get("AU", []),
#                 "abstract": record.get("AB", ""),
#                 "url": f"https://pubmed.ncbi.nlm.nih.gov/{record.get('PMID', '')}/",
#                 "published": record.get("DP", "")
#             })
#         return results

# class TavilyTool(BaseTool):
#     name = "tavily_search"
#     description = "Search the web using Tavily"
    
#     def __init__(self):
#         super().__init__()
#         self.tavily = TavilySearchResults()
    
#     def _run(self, query: str) -> List[Dict[str, Any]]:
#         return self.tavily.run(query)

def tavily_search_tool(search_query: str):
    """ Retrieve docs from web search with Tavily """
    print(f"Tavily search tool called with query: {search_query}")
    tavily_search = TavilySearchResults(max_results=5)
    # Search
    search_docs = tavily_search.invoke(search_query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    print(f'Tavily search returned: {formatted_search_docs}')
    return formatted_search_docs

# class VisualQATool(BaseTool):
#     name = "visual_qa"
#     description = "Analyze and understand images, graphs, and charts"
    
#     def __init__(self):
#         super().__init__()
#         self.llm = ChatOpenAI(model="gpt-4-vision-preview")
#         self.prompt = ChatPromptTemplate.from_messages([
#             ("system", "You are an expert at analyzing visual content. Describe what you see in detail, including any written text within the image.."),
#             ("user", "{image_url}")
#         ])
    
#     def _run(self, image_url: str) -> str:
#         chain = self.prompt | self.llm
#         return chain.invoke({"image_url": image_url})

# class FileReaderTool(BaseTool):
#     name = "file_reader"
#     description = "Read and parse PDFs and web pages"
    
#     def _run(self, file_path: str) -> str:
#         if file_path.endswith('.pdf'):
#             loader = PyPDFLoader(file_path)
#         else:
#             loader = WebBaseLoader(file_path)
        
#         docs = loader.load()
#         return "\n\n".join(doc.page_content for doc in docs)

# Export all tools
# RESEARCH_TOOLS = [
#     ArxivTool(),
#     PubMedTool(),
#     TavilyTool(),
#     VisualQATool(),
#     FileReaderTool()
# ] 
RESEARCH_TOOLS = [
    arxiv_search_tool,
    tavily_search_tool
]