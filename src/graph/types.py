import operator
from typing import List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


# ─────────────────────────────────────────────────────────────────────────────
# Researcher and Research Team Schemas
# ─────────────────────────────────────────────────────────────────────────────

class Researcher(BaseModel):
    role: str = Field(description="Primary role of the researcher")
    name: str = Field(description="Name of the researcher")
    domain: str = Field(description="Research focus or domain of the researcher as it relates to the topic")
    topic: str = Field(description="Research topic")
    initial_draft: str = Field(description="Initial draft of the research team")

    @property
    def persona(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Role: {self.role}\n"
            f"Domain: {self.domain}\n"
            f"Topic: {self.topic}\n"
            f"Initial Draft of team: {self.initial_draft}"
        )


class ResearcherList(BaseModel):
    researchers: List[Researcher] = Field(
        description="Comprehensive list of researchers with their roles and knowledge domain."
    )


# ─────────────────────────────────────────────────────────────────────────────
# State for Graph: create_researchers → feedback
# ─────────────────────────────────────────────────────────────────────────────

class ResearcherState(TypedDict):
    topic: str  # Research topic
    draft: str  # Human-written draft or knowledge base
    feedback: str  # Human-written feedback on research team design
    researchers: List[Researcher]


# ─────────────────────────────────────────────────────────────────────────────
# Subgraph State for Each Researcher
# ─────────────────────────────────────────────────────────────────────────────

class ResearchState(MessagesState):
    max_research_turns: int  # Max number of search-query turns
    context: Annotated[list, operator.add]  # Retrieved content
    researcher: Researcher  # Researcher object
    search_queries: Annotated[list, operator.add]  # List of queries used
    findings: Annotated[list, operator.add]  # Final output memo


# ─────────────────────────────────────────────────────────────────────────────
# Top-Level Graph State (Full research project)
# ─────────────────────────────────────────────────────────────────────────────

class ResearchGraphState(TypedDict):
    topic: str
    draft: str
    feedback: str
    researchers: List[Researcher]
    findings: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str


# ─────────────────────────────────────────────────────────────────────────────
# Structured Output Schema for Search Query
# ─────────────────────────────────────────────────────────────────────────────

class SearchQuery(BaseModel):
    search_query: str = Field(description="Search query for search tools")
